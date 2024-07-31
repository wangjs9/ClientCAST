import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import List
import numpy as np

from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..message import SYSTEM_NAME as SYSTEM
from ..message import Message
from .base import IntelligenceBackend, register_backend


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull."""
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


with suppress_stdout_stderr():
    # Try to import the transformers package
    try:
        import torch
        import transformers
        from transformers import pipeline
        from transformers.pipelines.conversational import (
            Conversation,
            ConversationalPipeline,
        )

        transformers.logging.set_verbosity_error()

    except ImportError:
        is_transformers_available = False
    else:
        is_transformers_available = True
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
except ImportError:
    is_transformers_available = False


@register_backend
class TransformersConversational(IntelligenceBackend):
    """Interface to the Transformers ConversationalPipeline."""

    stateful = False
    type_name = "transformers:conversational"

    def __init__(self, model: str, device: int = -1, **kwargs):
        super().__init__(model=model, device=device, **kwargs)
        self.model = model
        self.device = device
        assert is_transformers_available, "Transformers package is not installed"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

        if kwargs.pop('two_models', False):
            self.two_model = True
            self.tokenizer.pad_token = self.tokenizer.eos_token
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens", None)
            torch_dtype = kwargs["torch_dtype"] if "torch_dtype" in kwargs else torch.float16
            pretrained_model = AutoModelForCausalLM.from_pretrained(self.model, torch_dtype=torch_dtype)
            if "mistral" in self.model.lower():
                pretrained_model.generation_config.pad_token_id = pretrained_model.generation_config.eos_token_id
            self.chatbot = pipeline(
                task="conversational", model=pretrained_model, tokenizer=self.tokenizer, device=self.device, **kwargs
            )
        else:
            self.two_model = False
            if "mistral" in self.model.lower():
                self.chatbot = LLM(
                    model=self.model,
                    gpu_memory_utilization=.95,
                    # max_model_len=29534,
                    tensor_parallel_size=torch.cuda.device_count(),
                    disable_custom_all_reduce=True,
                    enforce_eager=True
                )
            else:
                self.chatbot = LLM(
                    model=self.model,
                    tensor_parallel_size=torch.cuda.device_count(),
                    disable_custom_all_reduce=True,
                    enforce_eager=True
                )
            self.sampling_params = SamplingParams(**kwargs)

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, conversation):
        if self.two_model:
            conversation = self.chatbot(conversation)
            response = conversation.generated_responses[-1]
        else:
            prompt_token_ids = self.tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, return_tensors="np").tolist()
            if len(prompt_token_ids) > 0 and isinstance(prompt_token_ids[0], type(np.array([]))):
                prompt_token_ids = [token_ids.tolist() for token_ids in prompt_token_ids]
            conversation = self.chatbot.generate(
                prompt_token_ids=prompt_token_ids, sampling_params=self.sampling_params, use_tqdm=False)
            if len(conversation) == 1:
                response = conversation[0].outputs[0].text
            else:
                response = [conv.outputs[0].text for conv in conversation]
        return response

    @staticmethod
    def _msg_template(agent_name, content):
        return f"[{agent_name}]: {content}"

    def query(
            self,
            agent_name: str,
            role_desc: str,
            history_messages: List[Message],
            global_prompt: str = None,
            request_msg: Message = None,
            *args,
            **kwargs,
    ) -> str:
        user_inputs, generated_responses = [], []
        all_messages = (
            [(SYSTEM, global_prompt), (SYSTEM, role_desc)]
            if global_prompt
            else [(SYSTEM, role_desc)]
        )

        for msg in history_messages:
            all_messages.append((msg.agent_name, msg.content))
        if request_msg:
            all_messages.append((SYSTEM, request_msg.content))

        prev_is_user = False  # Whether the previous message is from the user
        for i, message in enumerate(all_messages):
            if i == 0:
                assert (
                        message[0] == SYSTEM
                )  # The first message should be from the system

            if message[0] != agent_name:
                if not prev_is_user:
                    user_inputs.append(self._msg_template(message[0], message[1]))
                else:
                    user_inputs[-1] += "\n" + self._msg_template(message[0], message[1])
                prev_is_user = True
            else:
                if prev_is_user:
                    generated_responses.append(message[1])
                else:
                    generated_responses[-1] += "\n" + message[1]
                prev_is_user = False

        assert len(user_inputs) == len(generated_responses) + 1
        past_user_inputs = user_inputs[:-1]
        new_user_input = user_inputs[-1]

        # Recreate a conversation object from the history messages
        conversation = Conversation(
            text=new_user_input,
            past_user_inputs=past_user_inputs,
            generated_responses=generated_responses,
        )

        # Get the response
        response = self._get_response(conversation)
        return response

# conversation = Conversation("Going to the movies tonight - any suggestions?")
#
# # Steps usually performed by the model when generating a response:
# # 1. Mark the user input as processed (moved to the history)
# conversation.mark_processed()
# # 2. Append a mode response
# conversation.append_response("The Big lebowski.")
#
# conversation.add_user_input("Is it good?")
