import time
from tqdm import tqdm
import os
from transformers import BitsAndBytesConfig, pipeline, AutoTokenizer
import torch
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))
from baseline.generation_utils import *
from baseline.llm_prompting import *


def run_model(
    image_paths,
    task,
    ground_truth,
    results_json,
    map_manipulated,
    modality,
    model,
    evidence=[],
    evidence_idx=[],
    demonstrations=[],
    client=None,
    max_tokens=50,
    temperature=0.2,
    sleep=5,
    forgery_detection_explanation=None,
    forgery_detection_img_path=None,
):
    """
    Main loop to perform question answering with LLMs.
    Params:
        image_paths (list): a list containing the paths to the images
        question (str): the task to perform. One of [source, location, motivation]
        ground_truth (str): the ground truth answer, stored together with the prediction
        results_json (json): the json file where the model output should be saved
        map_manipulated (dict): a dictionary that maps manipulated images to their unaltered retrieved version.
        modality (str): the input modality to provide to the model. One of [vision, evidence, multimodal]
        model (str): the model to use. One of [gpt4, llava, llama]
        evidence (list): a list of dictionaries containing all the evidence
        evidence_idx (list): the index of the evidence to use
        demonstrations (list): a list of demonstrations for few-shot answer generation
        client  (object): the Azure OpenAI client object. Only required for gpt4 predictions
        max_tokens (int): the maximum number of tokens to generate as output
        temperature (float): the temperature of the model. Lower values make the output more deterministic
        sleep (int): the waiting time between two answer generation
        forgery_detection_explanation: explanation of detected forgery areas
        forgery_detection_img_path: stored image path
    """
    if model == "llava":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        model_id = "llava-hf/llava-1.5-7b-hf"
        pipe = pipeline(
            "image-to-text",
            model=model_id,
            model_kwargs={"quantization_config": quantization_config},
        )

    if model == "llama":
        model_id = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    # Select the prompt assembler
    prompt_assembler_dict = {
        "gpt4": assemble_prompt_gpt4,
        "llava": assemble_prompt_llava,
        "llama": assemble_prompt_llama,
    }
    assembler = prompt_assembler_dict[model]

    questions = {
        "source": "Who is the source/author of this image? Answer only with one or more persons or entities in a few words.",
        "date": "When was this image taken? Answer only with one or more dates in a few words.",
        "location": "Where was this image taken? Answer only with one or more locations in a few words.",
        "motivation": "Why was this image taken? Answer in a few words.",
    }

    question = questions[task]

    # Main loop
    for i in tqdm(range(len(image_paths))):
        if modality in ["evidence", "multimodal"]:
            if len(evidence_idx[i]) != 0:
                # Take the subset of evidence matching the image, then take the top K based on CLIP ranking
                evidence_image_subset = [
                    ev for ev in evidence if ev["image path"] == image_paths[i]
                ]
                evidence_selection = [
                    evidence_image_subset[idx] for idx in evidence_idx[i]
                ]

                prompt = assembler(
                    question, evidence=evidence_selection, modality=modality
                )
            else:
                prompt = assembler(question, modality="vision")
        if modality == "evidence":
            if len(evidence_idx[i]) == 0:
                print("No evidence")
                output = ""
            else:
                if model == "gpt4":
                    output = gpt4_vision_prompting(
                        prompt,
                        client,
                        image_paths[i],
                        map_manipulated,
                        modality=modality,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                elif model == "llava":
                    output = llava_prompting(
                        prompt,
                        image_paths[i],
                        pipe,
                        map_manipulated,
                        temperature,
                        max_tokens,
                    )
                elif model == "lama":
                    output = llama_prompting(
                        prompt, pipe, tokenizer, temperature, max_tokens
                    )

                else:
                    print("Error : wrong model provided")
                    break
        else:
            if model == "gpt4":
                output = gpt4_vision_prompting(
                    prompt,
                    client,
                    image_paths[i],
                    modality=modality,
                    map_manipulated_original=map_manipulated,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # NOTE: Uncomment these lines if you want to access forgery detection performance
                # output = gpt4_vision_prompting_with_forgery_detection(prompt,
                #                                client,
                #                                image_paths[i],
                #                                modality=modality,
                #                                map_manipulated_original=map_manipulated,
                #                                temperature=temperature,
                #                                max_tokens=max_tokens,
                #                                forgery_detection_explanation= forgery_detection_explanation[image_paths == image_name]['outputs'],
                #                                forgery_detection_path=forgery_detection_img_path + '/' + image_name,
                #                                )
            elif model == "llava":
                output = llava_prompting(
                    prompt,
                    image_paths[i],
                    pipe,
                    map_manipulated,
                    temperature,
                    max_tokens,
                )
            else:
                print("Error : wrong model provided")
                break
        # Save results
        if type(output) == str:
            data = {}
            data["img_path"] = image_paths[i]
            data["ground_truth"] = ground_truth[i]
            data["output"] = output
            save_result(data, results_json)
            time.sleep(sleep)


def run_core_model(
    image_paths,
    task,
    ground_truth,
    results_json,
    map_manipulated,
    modality,
    model,
    storys,
    evidence=[],
    evidence_idx=[],
    confidence_levels=[],
    demonstrations=[],
    client=None,
    max_tokens=50,
    temperature=0.2,
    sleep=5,
    forgery_detection_explanation=None,
    forgery_detection_img_path=None,
):
    """
    Main loop to perform question answering with LLMs.
    Params:
        image_paths (list): a list containing the paths to the images
        question (str): the task to perform. One of [source, location, motivation]
        ground_truth (str): the ground truth answer, stored together with the prediction
        results_json (json): the json file where the model output should be saved
        map_manipulated (dict): a dictionary that maps manipulated images to their unaltered retrieved version.
        modality (str): the input modality to provide to the model. One of [vision, evidence, multimodal]
        model (str): the model to use. One of [gpt4, llava, llama]
        evidence (list): a list of dictionaries containing all the evidence
        evidence_idx (list): the index of the evidence to use
        demonstrations (list): a list of demonstrations for few-shot answer generation
        client  (object): the Azure OpenAI client object. Only required for gpt4 predictions
        max_tokens (int): the maximum number of tokens to generate as output
        temperature (float): the temperature of the model. Lower values make the output more deterministic
        sleep (int): the waiting time between two answer generation
        forgery_detection_explanation: explanation of detected forgery areas
        forgery_detection_img_path: stored image path
    """
    if model == "llava":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        model_id = "llava-hf/llava-1.5-7b-hf"
        pipe = pipeline(
            "image-to-text",
            model=model_id,
            model_kwargs={"quantization_config": quantization_config},
        )

    if model == "llama":
        model_id = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    # Select the prompt assembler
    prompt_assembler_dict = {
        "gpt4": assemble_prompt_gpt4,
        "llava": assemble_prompt_llava,
        "llama": assemble_prompt_llama,
    }
    assembler = prompt_assembler_dict[model]

    # Main loop
    for i in tqdm(range(len(image_paths))):
        filename = os.path.basename(image_paths[i])
        filename_without_ext = os.path.splitext(filename)[0]
        # meaningful_part = filename_without_ext.split("-is-shared-as")[0]
        meaningful_part = filename_without_ext.split("-in-")[0]
        questions = {
            "source": f"Who is the source/author of this image? Answer only with one or more persons or entities in a few words. Please refer to the input, if there is more than one source then select the most relevant part to “{meaningful_part}” and generate an answer, even if you can not be sure, please select the most likely source.",
            "date": f"When was this photo taken? Please refer to the date value in the input, if there is more than one date then select the most relevant part to “{meaningful_part}” and generate an answer, please only answer the date, even if you can not be sure, please select the most likely date.",
            "location": f"Where was this image taken? Answer only with one or more locations in a few words.Please refer to the input, if there is more than one location then select the most relevant part to “{meaningful_part}” and generate an answer, even if you can not be sure, please select the most likely location.Please only answer the location.",
            "motivation": f"Why was this image taken? Answer in a few words.Please refer to the input, if there is more than one motivation then select the most relevant part to “{meaningful_part}” and generate an answer, even if you can not be sure, please select the most likely motivation.",
        }

        question = questions[task]

        if modality in ["evidence", "multimodal"]:
            if len(evidence_idx[i]) != 0:
                # Take the subset of evidence matching the image, then take the top K based on CLIP ranking
                evidence_image_subset = [
                    ev for ev in evidence if ev["image path"] == image_paths[i]
                ]
                evidence_selection = [
                    evidence_image_subset[idx] for idx in evidence_idx[i]
                ]

                prompt = assembler(
                    question, evidence=evidence_selection, modality=modality
                )
            else:
                prompt = assembler(question, modality="vision")
        if modality == "evidence":
            if len(evidence_idx[i]) == 0:
                print("No evidence")
                output = ""
            else:
                if model == "gpt4":
                    output = gpt4_vision_prompting(
                        prompt,
                        client,
                        image_paths[i],
                        map_manipulated,
                        modality=modality,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                elif model == "llava":
                    output = llava_prompting(
                        prompt,
                        image_paths[i],
                        pipe,
                        map_manipulated,
                        temperature,
                        max_tokens,
                    )
                elif model == "lama":
                    output = llama_prompting(
                        prompt, pipe, tokenizer, temperature, max_tokens
                    )

                else:
                    print("Error : wrong model provided")
                    break
        else:
            if model == "gpt4":
                output = gpt4_vision_prompting(
                    prompt,
                    client,
                    image_paths[i],
                    modality=modality,
                    map_manipulated_original=map_manipulated,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            # NOTE: Uncomment these lines if you want to access forgery detection performance
            # image_name = image_paths[i].split('/')[-1]
            # output = gpt4_vision_prompting_with_forgery_detection(prompt,
            #                                 client,
            #                                 image_paths[i],
            #                                 modality=modality,
            #                                 map_manipulated_original=map_manipulated,
            #                                 temperature=temperature,
            #                                 max_tokens=max_tokens,
            #                                 forgery_detection_explanation= forgery_detection_explanation[image_paths == image_name]['outputs'],
            #                                 forgery_detection_path=forgery_detection_img_path + '/' + image_name,
            #                                 )

            elif model == "llava":
                output = llava_prompting(
                    prompt,
                    image_paths[i],
                    pipe,
                    map_manipulated,
                    temperature,
                    max_tokens,
                )
            else:
                print("Error : wrong model provided")
                break
        # Save results
        if type(output) == str:
            data = {}
            data["img_path"] = image_paths[i]
            data["ground_truth"] = ground_truth[i]
            data["output"] = output
            data["confidence"] = confidence_levels[i]
            save_result(data, results_json)
            time.sleep(sleep)


def run_fallback_model(
    image_paths,
    task,
    ground_truth,
    results_json,
    map_manipulated,
    model,
    evidence=[],
    evidence_idx=[],
    story_evidence=[],
    confidence_levels=[],
    client=None,
    max_tokens=50,
    temperature=0.2,
    sleep=5,
):
    """
    Main loop to perform question answering with LLMs.
    Params:
        image_paths (list): a list containing the paths to the images
        question (str): the task to perform. One of [source, location, motivation]
        ground_truth (str): the ground truth answer, stored together with the prediction
        results_json (json): the json file where the model output should be saved
        map_manipulated (dict): a dictionary that maps manipulated images to their unaltered retrieved version.
        model (str): the model to use. One of [gpt4, llava, llama]
        evidence (list): a list of dictionaries containing all the evidence
        evidence_idx (list): the index of the evidence to use
        story_evidence (list): a list of story evidence
        confidence_levels (list): a list of confidence levels
        client  (object): the Azure OpenAI client object. Only required for gpt4 predictions
        max_tokens (int): the maximum number of tokens to generate as output
        temperature (float): the temperature of the model. Lower values make the output more deterministic
        sleep (int): the waiting time between two answer generation.
    """

    if model == "llava":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        model_id = "llava-hf/llava-1.5-7b-hf"
        pipe = pipeline(
            "image-to-text",
            model=model_id,
            model_kwargs={"quantization_config": quantization_config},
        )

    # Select the prompt assembler
    prompt_assembler_dict = {
        "gpt4": assemble_prompt_gpt4,
        "llava": assemble_prompt_llava,
    }
    assembler = prompt_assembler_dict[model]

    questions = {
        "source": "Who is the source/author of this image? Answer only with one or more persons or entities in a few words.",
        "date": "When was this image taken? Answer only with one or more dates in a few words.",
        "location": "Where was this image taken? Answer only with one or more locations in a few words.",
        "motivation": "Why was this image taken? Answer in a few words.",
    }

    question = questions[task]

    # Main loop
    for i in tqdm(range(len(image_paths))):
        story_based_evidence = story_evidence.get(i, None)
        if story_based_evidence:
            print(f"{i+1}. prediction has used STORY!")
            story_evidence_subset = [
                ev
                for ev in evidence
                if ev["image path"] == image_paths[i] and ev["downloaded"]
            ]
            story_evidence_selection = [
                story_evidence_subset[idx] for idx in story_based_evidence
            ]
            prompt = assembler(
                question, evidence=story_evidence_selection, modality="story"
            )
        else:
            if len(evidence_idx[i]) != 0:
                # Take the subset of evidence matching the image, then take the top K based on CLIP ranking
                evidence_image_subset = [
                    ev
                    for ev in evidence
                    if ev["image path"] == image_paths[i] and ev["downloaded"]
                ]
                evidence_selection = [
                    evidence_image_subset[idx] for idx in evidence_idx[i]
                ]
                prompt = assembler(
                    question, evidence=evidence_selection, modality="multimodal"
                )
            else:
                prompt = assembler(question, modality="vision")

        if model == "gpt4":
            output = gpt4_vision_prompting(
                prompt,
                client,
                image_paths[i],
                map_manipulated_original=map_manipulated,
                modality="multimodal",
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif model == "llava":
            output = llava_prompting(
                prompt, image_paths[i], pipe, map_manipulated, temperature, max_tokens
            )
        else:
            print("Error : wrong model provided")
            break
        # Save results
        if isinstance(output, str):
            data = {}
            data["img_path"] = image_paths[i]
            data["ground_truth"] = ground_truth[i]
            data["output"] = output
            # Add indicator of using story for generating process
            if story_based_evidence:
                data["STORY"] = True
            else:
                data["STORY"] = False
            data["confidence"] = confidence_levels[i]
            save_result(data, results_json)
            time.sleep(sleep)
