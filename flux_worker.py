import logging
import time
import datetime
import argparse
import os
import io
import math
import random
import torch
import torch.distributed
from torch import Tensor
from transformers import T5EncoderModel
from xfuser import xFuserFluxPipeline, xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
    get_pipeline_parallel_world_size,
    get_classifier_free_guidance_world_size,
    get_tensor_model_parallel_world_size,
    get_data_parallel_world_size,
)
from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps, calculate_shift

from PIL import Image
import numpy as np

from aime_api_worker_interface import APIWorkerInterface

WORKER_JOB_TYPE = "flux_dev"
DEFAULT_WORKER_AUTH_KEY = "2a14da16a70713bb3a4484b4ae5f681f"
VERSION = 2

class Inferencer():
    """
    The Inferencer class is responsible for loading models and generating images
    using the loaded models. It handles the configuration of the model pipeline
    and manages the inference process.
    """

    engine_args = None
    engine_config = None

    def __init__(self, ckpt_dir: str, world_size:int):
        """
        Initialize the Inferencer with the given checkpoint directory and world size.

        Args:
            ckpt_dir (str): Directory containing the model checkpoint.
            world_size (int): The number of processes participating in the distributed training.
        """
        self.engine_args = xFuserArgs(ckpt_dir)

        # pipefusion parallel
        self.engine_args.dit_parallel_size = world_size
        self.engine_args.pipefusion_parallel_degree = 1
        # sequence parallel
        self.engine_args.ulysses_degree = world_size
        self.engine_args.ring_degree = 1
        # tensor parallel
        self.engine_args.tensor_parallel_degree = 1
        self.engine_args.split_scheme = "row"
        # data parallel (batch)
        self.engine_args.data_parallel_degree = 1

        self.engine_args.use_parallel_vae = False
        self.engine_args.use_fp8_t5_encode = True
        self.engine_args.use_torch_compile = False

        self.engine_config, self.input_config = self.engine_args.create_config()
        self.engine_config.runtime_config.dtype = torch.bfloat16
        self.local_rank = get_world_group().local_rank
        self.text_encoder_2 = None
        self.pipe = None

    def load_models(self):
        """
        Load the models required for image generation. This includes loading
        the text encoder and setting up the pipeline for inference.
        """
        self.text_encoder_2 = T5EncoderModel.from_pretrained(self.engine_config.model_config.model, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)

        if self.engine_args.use_fp8_t5_encoder:
            from optimum.quanto import freeze, qfloat8, quantize
            logging.info(f"rank {local_rank} quantizing text encoder 2")
            quantize(self.text_encoder_2, weights=qfloat8)
            freeze(self.text_encoder_2)

        cache_args = {
                "use_teacache": False,
                "use_fbcache": True,
                "rel_l1_thresh": 0.12,
                "return_hidden_states_first": False,
                "num_steps": 50,
            }

        self.pipe = xFuserFluxPipeline.from_pretrained(
            pretrained_model_name_or_path=self.engine_config.model_config.model,
            engine_config=self.engine_config,
            cache_args=cache_args,
            torch_dtype=torch.bfloat16,
            text_encoder_2=self.text_encoder_2,
        )

        self.pipe = self.pipe.to(f"cuda:{self.local_rank}")

        self.pipe.prepare_run(self.input_config, steps=1)

    def gen_noise(
        self,
        num_samples: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
    ):
        return torch.randn(
            num_samples,
            16,
            # allow for packing
            2 * math.ceil(height / 16),
            2 * math.ceil(width / 16),
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(seed),
        )


    def precalc_timesteps(self, latents, num_inference_steps):
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.pipe.scheduler.config.base_image_seq_len,
            self.pipe.scheduler.config.max_image_seq_len,
            self.pipe.scheduler.config.base_shift,
            self.pipe.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.pipe.scheduler,
            num_inference_steps,
            device=None,
            timesteps=None,
            sigmas=sigmas,
            mu=mu,
        )
        return timesteps


    def gen_image(self, prompt, callback, width, height, num_steps, seed, guidance, progress_images=False, init_image=None, image2image_strength=0.8):
        """
        Generate an image based on the given prompt and parameters.

        Args:
            prompt (str): The text prompt used to generate the image.
            callback (function): The callback function to process the output image.
            width (int): The width of the generated image.
            height (int): The height of the generated image.
            num_steps (int): The number of inference steps to perform.
            seed (int): The random seed for reproducibility.
            guidance (float): The guidance scale for the image generation.
            progress_images (bool, optional): Whether to provide progress images. Defaults to False.
            init_image (PIL.Image, optional): The initial image for image-to-image generation. Defaults to None.
            image2image_strength (float, optional): The strength for image-to-image generation. Defaults to 0.8.

        Returns:
            PIL.Image: The generated image.
        """
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        progress_step = 1
        if self.local_rank == 0:
            callback(None, progress_step, False, message='Preparing image...')

        self.input_config.height = height
        self.input_config.width = width
        self.pipe.reset_activation_cache()
        self.pipe.reset_transformer_cache()

        def pipe_callback_on_step_end(pipeline, step: int, timestep: int, callback_kwargs):
            if self.local_rank == 0:
                callback(None, 1 + ((num_steps - 2) * step) / num_steps, False, message='Denoising...')
            return callback_kwargs

        latents = None
        generator = None

        noise_latents = self.gen_noise(
               1,
               height,
               width,
               device=None,
               dtype=torch.float,
               seed=seed
        ).cuda()

        timestep_offset = 0

        if init_image: 
            init_image = init_image.convert("RGB")
            init_image = np.array(init_image).astype(np.float32)
            init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 128.0
            init_image = init_image - 1.0; 
            init_image = init_image.unsqueeze(0) 
            init_image = torch.nn.functional.interpolate(init_image, (height, width))
            latents = self.pipe.vae.encode(init_image.bfloat16().cuda())["latent_dist"].sample() 
            latents = latents - self.pipe.vae.config.shift_factor
            latents = latents * self.pipe.vae.config.scaling_factor  

            vae_height = 2 * (int(height) // (self.pipe.vae_scale_factor * 2))
            vae_width = 2 * (int(width) // (self.pipe.vae_scale_factor * 2))            

            platents = self.pipe._pack_latents(latents.bfloat16(), 1, 16, vae_height, vae_width)
            
            timesteps = self.precalc_timesteps(platents, num_steps)

            timestep_offset = int((1 - image2image_strength) * num_steps)
            t = timesteps[timestep_offset] * 0.001
            latents = t * noise_latents + (1.0 - t) * latents.float()

            latents = self.pipe._pack_latents(latents.bfloat16(), 1, 16, vae_height, vae_width)
        else:
            generator = torch.Generator(device="cuda").manual_seed(seed)

        output = self.pipe(
            height=height,
            width=width,
            prompt=prompt,
            num_warmup_steps=1,
            num_inference_steps=num_steps,
            output_type="pil",
            max_sequence_length=512,
            guidance_scale=guidance,
            generator=generator,
            latents=latents,
            timesteps = None,
            timestep_offset=timestep_offset,
            callback_on_step_end=pipe_callback_on_step_end
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{self.local_rank}")

        dp_group_index = get_data_parallel_rank()
        num_dp_groups = get_data_parallel_world_size()
        dp_batch_size = (self.input_config.batch_size + num_dp_groups - 1) // num_dp_groups

        last_rank = get_world_group().world_size - 1

        images = []

        if self.pipe.is_dp_last_group():
            for i, image in enumerate(output.images):
                images.append(image)
        else:
            images.append(Image.new("RGB", (width, height), (255, 255, 255)))

        torch.distributed.broadcast_object_list(images, last_rank)

        if(self.local_rank == 0):
            callback(images[0])
            print(
                f"epoch time: {elapsed_time:.2f} sec, peak GPU memory: {peak_memory/1e9:.2f} GB"
            )

class ProcessOutputCallback():
    """
    The ProcessOutputCallback class is responsible for processing the output
    of the image generation and sending the results back to the API worker.
    It handles progress updates and final results.
    """

    def __init__(self, api_worker, inferencer, model_name):
        """
        Initialize the ProcessOutputCallback with the given API worker, inferencer, and model name.

        Args:
            api_worker (APIWorkerInterface): The API worker interface to send results.
            inferencer (Inferencer): The inferencer instance used for image generation.
            model_name (str): The name of the model used for inference.
        """
        self.api_worker = api_worker
        self.inferencer = inferencer
        self.model_name = model_name
        self.job_data = None
        self.arrival_time = None
        self.finished_time = None
        self.preprocessing_duration = None

    def process_output(self, image, progress_step=100, finished=True, error=None, message=None):
        """
        Process the output image and send the results to the API worker.

        Args:
            image (PIL.Image): The generated image.
            progress_step (int, optional): The current progress step. Defaults to 100.
            finished (bool, optional): Whether the generation is finished. Defaults to True.
            error (str, optional): Error message if any. Defaults to None.
            message (str, optional): Additional message to include in the progress. Defaults to None.
        """
        if error:
            print('error')
            self.api_worker.send_progress(100, None)
            image = Image.fromarray((np.random.rand(1024,1024,3) * 255).astype(np.uint8))
            return self.api_worker.send_job_results({
                'images': [image],
                'error': error,
                'model_name': self.model_name
            })
        else:
            if not finished:
                step_factor = self.job_data.get('image2image_strength') if self.job_data.get('image') else 1
                total_steps = int(self.job_data.get('steps') * step_factor) + 3
                progress_info = round((progress_step) * 100 / total_steps)

                if self.api_worker.progress_data_received:
                    progress_data = {'progress_message': message}
                    if image is not None:
                        progress_data['progress_images'] = [image]
                    return self.api_worker.send_progress(progress_info, progress_data)
            else:
                self.finished_time = time.time()

                image_list = [image]
                self.api_worker.send_progress(100, None)
                return self.api_worker.send_job_results({
                    'images': image_list,
                    'seed': self.job_data.get('seed'),
                    'model_name': self.model_name,
                    "finished_time": self.finished_time,
                    "arrival_time": self.arrival_time,
                    "preprocessing_duration": self.preprocessing_duration,
                    "metrics": {
                        "out_num_images": len(image_list),
                        "out_resolution": (self.job_data.get("width"), self.job_data.get("height"))
                    }
                })

def load_flags():
    """
    Load command-line arguments using argparse.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api_server", type=str, default="http://0.0.0.0:7777", help="Address of the AIME API server"
                        )
    parser.add_argument(
        "--gpu_id", type=int, default=0, required=False, help="ID of the GPU to be used"
                        )
    parser.add_argument(
        "--ckpt_dir", type=str, default="/models/FLUX.1-dev/", help="Destination of model weigths"
                        )
    parser.add_argument(
        "--api_auth_key", type=str , default=DEFAULT_WORKER_AUTH_KEY, required=False,
        help="API server worker auth key",
    )
    return parser.parse_args()

def convert_binary_to_image(image_data, width, height):
    """
    Convert binary image data to a PIL Image object and resize it.

    Args:
        image_data (bytes): The binary image data.
        width (int): The desired width of the image.
        height (int): The desired height of the image.

    Returns:
        PIL.Image: The resized image.
    """
    if image_data:
        with io.BytesIO(image_data) as buffer:
            image = Image.open(buffer)
            return image.resize((width, height), Image.LANCZOS)

@torch.no_grad()
def main():
    """
    The main function that sets up the API worker, loads models, and processes jobs.
    This is the entry point for the worker process.
    """
    args = load_flags()
    device = "cuda:" + str(args.gpu_id)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # torch.set_default_device(device)
    api_worker = APIWorkerInterface(
        args.api_server, WORKER_JOB_TYPE, args.api_auth_key, args.gpu_id,
        world_size=world_size, rank=local_rank,
        gpu_name=torch.cuda.get_device_name(), worker_version=VERSION
    )

    print("Loading models... ")
    inferencer = Inferencer(args.ckpt_dir, world_size)
    inferencer.load_models()

    callback = ProcessOutputCallback(api_worker, inferencer, 'flux-dev')

    print("Waiting for jobs... ")

    while True:
        try:
            callback.arrival_time = time.time()

            job_data = api_worker.job_batch_request(1)

            init_image = [None]

            if local_rank == 0:
                job_data = job_data[0]
                print(f'Processing job {job_data.get("job_id")}...', end='', flush=True)

                preprocessing_start = time.time()

                seed = job_data.get('seed', -1)
                if seed == -1:
                    random.seed(datetime.datetime.now().timestamp())
                    seed = random.randint(1, 99999999)
                    job_data['seed'] = seed

                seed = [seed]
                init_image_data = api_worker.get_binary(job_data, 'image')
                if init_image_data:
                    init_image = [convert_binary_to_image(
                        init_image_data,
                        job_data.get('width'),
                        job_data.get('height')
                    )]

                callback.job_data = job_data

                callback.preprocessing_duration = time.time() - preprocessing_start
                prompt = [job_data.get('prompt')]
                width = [job_data.get('width')]
                height = [job_data.get('height')]
                steps = [job_data.get('steps')]
                guidance = [job_data.get('guidance')]
                progress_images = [job_data.get('provide_progress_images') == "decoded"]
                image2image_strength = [job_data.get('image2image_strength')]
            else:
                seed = [0]
                callback.job_data = job_data
                prompt = [""]
                width = [1024]
                height = [1024]
                steps = [0]
                guidance = [1.0]
                progress_images = [False]
                image2image_strength = [0.0]

            torch.distributed.broadcast_object_list(prompt, 0)
            torch.distributed.broadcast_object_list(seed, 0)
            has_init_image = [init_image[0] != None]
            torch.distributed.broadcast_object_list(has_init_image, 0)
            if has_init_image[0]:
                torch.distributed.broadcast_object_list(init_image, 0)
            torch.distributed.broadcast_object_list(width, 0)
            torch.distributed.broadcast_object_list(height, 0)
            torch.distributed.broadcast_object_list(steps, 0)
            torch.distributed.broadcast_object_list(guidance, 0)
            torch.distributed.broadcast_object_list(progress_images, 0)
            torch.distributed.broadcast_object_list(image2image_strength, 0)

            image = inferencer.gen_image(
                prompt[0],
                callback.process_output,
                width[0],
                height[0],
                steps[0],
                seed[0],
                guidance[0],
                progress_images[0],
                init_image[0],
                image2image_strength[0]
            )

            print('Done')

        except ValueError as exc:
            print('Error')
            callback.process_output(None, None, True, f'{exc}\nChange parameters and try again')
            continue
        except torch.cuda.OutOfMemoryError as exc:
            print('Error - CUDA OOM')
            callback.process_output(None, None, True, f'{exc}\nReduce image size and try again')
            continue
        except OSError as exc:
            print('Error')
            callback.process_output(None, None, True, f'{exc}\nInvalid image file')
            continue

    get_runtime_state().destroy_distributed_env()

if __name__ == "__main__":
    main()
