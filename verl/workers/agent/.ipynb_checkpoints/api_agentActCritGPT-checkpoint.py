import re
import logging
import ast
import json
import base64
import os
import requests
import numpy as np
from copy import deepcopy
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any, Tuple
from omegaconf import DictConfig
from .qwen_tools import fetch_tools
import time

logger = logging.getLogger(__name__)

class APIAgentAC:
    """
    Agent class for handling multi-turn action/observation interactions, using OpenAI API for conversation generation.
    Handles interaction between LLM and environment, supports multimodal observation.
    """
    def __init__(
        self,
        config: DictConfig,
    ):
        """
        Initialize Agent.
        
        Args:
            config: Configuration information
        """
        self.config = config
        self.tools = fetch_tools(placeholder='<image>')
        
        # Define regex for action detection
        self.tool_pattern = config.get("tool_pattern", r"<tool_call>(.*?)</tool_call>")
        self.image_placeholder = config.get("image_placeholder", "<image>")
        
        # Configure max turns and max tokens per turn
        self.max_turns = config.get("max_turns", 4)
        self.max_tokens_per_turn = config.get("max_tokens_per_turn", 1024)
        self.max_results = config.get("max_results", 3)
        
        # Whether to save intermediate responses
        self.save_intermediate_responses = config.get("save_intermediate_responses", False)
        
        # OpenAI API config
        self.openai_api_key = config.get("openai_api_key")
        self.openai_api_base = config.get("openai_api_base", "https://api.openai.com/v1/chat/completions")
        self.openai_model = config.get("openai_model", "gpt-4o-mini")
        self.openai_temperature = config.get("openai_temperature", 0.8)
        self.openai_top_p = config.get("openai_top_p", 0.9)

        #re-think config
        self.max_rethinks = config.get("max_rethinks", 2)
        self.rethink_count = 0
        self.last_boxed_answer = None
        self.wrong_answers = set()
        self.wrong_sum = set()
        self.code_switch = False

        # Per-attempt tool usage
        self.tool_used_in_attempt = False

    def process_action_and_execute(self, text: str, env_state: Dict[str, Any] = None) -> Tuple[bool, bool, Dict[str, Any]]:
        """
        Extracts action list from text and executes them one by one, returns merged result.
        Result text is wrapped with <observation> tag.

        Args:
            text: Generated text
            env_state: Environment state

        Returns:
            (has_action, all_actions_successful, result_dict):
            - has_action: Boolean, whether tool_call block found.
            - all_actions_successful: Boolean, whether all actions executed successfully.
            - result_dict: Dict containing merged observation result.
        """
        obs_template = "{image} \n <TOOL_RES>{text}\nNow think first again, then decide to call tools or answer.</TOOL_RES>"

        tool_match = re.search(self.tool_pattern, text, re.DOTALL)
        if not tool_match:
            tool_match = re.search(r'"tool_calls": (\[.*\])', text, re.DOTALL)
        if not tool_match:
            no_action_msg = obs_template.format(text="No action found.",image='')
            return False, False, {"text": no_action_msg, "image": None}

        try:
            tool_content_str = tool_match.group(1).strip()
            tool_calls_list = ast.literal_eval(tool_content_str)

            if not isinstance(tool_calls_list, list):
                if isinstance(tool_calls_list, dict):
                    tool_calls_list = [tool_calls_list]
                else:
                    raise ValueError("Tool content inside <tool_call> is not a valid list or dictionary of tool calls.")
            if len(tool_calls_list) == 0:
                return False, False, {"text": obs_template.format(text="No tool calls found in the list.",image=''), "image": None}

        except Exception as e:
            error_msg = obs_template.format(text=f"Error parsing tool calls list: {str(e)}",image='')
            return True, False, {"text": error_msg, "image": None}

        if not tool_calls_list:
            empty_list_msg = obs_template.format(text="Tool call list is empty.",image='')
            return True, False, {"text": empty_list_msg, "image": None}

        aggregated_obs_texts = []
        aggregated_images = []
        all_actions_successful = True

        for i, tool_content in enumerate(tool_calls_list[:self.max_results]):
            if not isinstance(tool_content, dict):
                error_msg = f"Invalid item in tool call list: Expected a dictionary, got {type(tool_content)} ({str(tool_content)[:100]})."
                aggregated_obs_texts.append(error_msg)
                all_actions_successful = False
                continue

            try:
                tool_name = tool_content.get('name')
                args = tool_content.get('arguments')

                if not tool_name:
                    error_msg = f"Missing 'name' in tool content: {str(tool_content)[:100]}"
                    aggregated_obs_texts.append(error_msg)
                    all_actions_successful = False
                    continue
                
                if args is None:
                    args = {}
                elif not isinstance(args, dict):
                    error_msg = f"Invalid 'arguments' for tool '{tool_name}': Expected a dictionary, got {type(args)}."
                    aggregated_obs_texts.append(error_msg)
                    all_actions_successful = False
                    continue

            except Exception as e:
                error_msg = f"Error parsing individual tool call '{str(tool_content)[:100]}': {str(e)}"
                aggregated_obs_texts.append(error_msg)
                all_actions_successful = False
                continue

            if tool_name not in self.tools:
                error_msg = f"Error: There is no tool named '{tool_name}'."
                aggregated_obs_texts.append(error_msg)
                all_actions_successful = False
                continue

            try:
                result = self.tools[tool_name].call(args, env_state)

                tool_obs_text = result.get('text', '')
                if tool_obs_text:
                    aggregated_obs_texts.append(f"Image {i+len(env_state['image'])}:" + str(tool_obs_text))

                tool_obs_image = result.get("image", None)
                if tool_obs_image is not None:
                    if isinstance(tool_obs_image, list):
                        aggregated_images.extend(tool_obs_image)
                    else:
                        aggregated_images.append(tool_obs_image)
            
            except Exception as e:
                error_msg = f"Failed to call tool '{tool_name}' with args '{str(args)[:100]}' due to error: {str(e)}"
                aggregated_obs_texts.append(error_msg)
                all_actions_successful = False

        final_obs_content = "\n".join(aggregated_obs_texts)
        if not final_obs_content and tool_calls_list:
            if all_actions_successful:
                final_obs_content = "All actions processed successfully with no textual output."
            else:
                final_obs_content = "Actions processed with errors, but no specific textual error messages were generated."
        elif not tool_calls_list:
             final_obs_content = "No actions to process."

        formatted_final_obs_text = obs_template.format(text=final_obs_content,image=self.image_placeholder*len(aggregated_images))
        
        final_images = aggregated_images if aggregated_images else None

        return True, all_actions_successful, {"text": formatted_final_obs_text, "image": final_images}

    def is_base64_string(self, s):
        """Check if string is valid base64 encoding"""
        if not isinstance(s, str):
            return False
        try:
            if "base64," in s:
                s = s.split("base64,")[1]
            base64.b64decode(s)
            return True
        except Exception:
            return False

    def _encode_image_to_base64(self, image):
        """Convert PIL image, numpy array, or base64 string to base64 encoding"""
        if isinstance(image, str) and self.is_base64_string(image):
            if "base64," in image:
                return image.split("base64,")[1]
            return image
            
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if isinstance(image, Image.Image):
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
            
        raise ValueError(f"Unsupported image type: {type(image)}")


    def _prepare_message_with_images(self, text, images=None):
        """
        Prepare mixed content of text and images.
        Handles <image> placeholders in text, inserts corresponding images.
        """
        content = []
        
        if "<image>" not in text or not images:
            if images and "<image>" not in text:
                for image in images:
                    if isinstance(image, str) and image.startswith("http"):
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": image}
                        })
                    else:
                        base64_image = self._encode_image_to_base64(image)
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        })
            content.append({"type": "text", "text": text})
            return content
        parts = text.split("<image>")
        image_count = len(parts) - 1
        
        if parts[0]:
            content.append({"type": "text", "text": parts[0]})
        
        for i in range(image_count):
            if i < len(images):
                image = images[i]
                if isinstance(image, str) and image.startswith("http"):
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": image}
                    })
                else:
                    base64_image = self._encode_image_to_base64(image)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    })
            
            if i+1 < len(parts) and parts[i+1]:
                content.append({"type": "text", "text": parts[i+1]})
        
        for i in range(image_count, len(images)):
            image = images[i]
            if isinstance(image, str) and image.startswith("http"):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image}
                })
            else:
                base64_image = self._encode_image_to_base64(image)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })
                
        return content

    def _call_openai_api(self, messages, max_tokens=None, temperature=None, stop=None, rformat=None):
        """Call OpenAI API for generation"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not provided")
        
        url = f"{self.openai_api_base}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }
        
        payload = {
            "model": self.openai_model,
            "messages": messages,
            "temperature": temperature or self.openai_temperature,
            "top_p": self.openai_top_p,
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
            
        if stop:
            payload["stop"] = stop
        if rformat:
            payload["response_format"]: rformat 
        
        max_retries = 1
        retry_count = 0
        print(f" Qwery model:{self.openai_model}")
        while retry_count < max_retries:
            try:
                response = requests.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                retry_count += 1
                logger.error(f"Error calling OpenAI API (attempt {retry_count}/{max_retries}): {str(e)}")
                #print(f" This is messages:{model}")

                
                if retry_count >= max_retries:
                    logger.error(f"Failed after {max_retries} attempts")
                    raise
                
                time.sleep(1)
    def extract_boxed_answer(self, text: str) -> str:
        """
        Extract option letter from <answer>...</answer>.
        Returns 'A'–'E' or None.
        """
        if not text:
            return None
    
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            text,
            re.DOTALL | re.IGNORECASE
        )
        search_text = answer_match.group(1) if answer_match else text
    
        boxed = re.search(r"\\boxed\{([A-E])\}", search_text)
        return boxed.group(1) if boxed else None
        
################################################################
# Critic start
###############################################################
    def handle_language_gpt(
    self,
    generated_text: str,
    conversation_history: List[Dict[str, Any]],
    gtl
    ) -> bool:
        """
        Trigger retry with English question if code-switch OR repetition detected.
        """
        
        lang_check=[]
        
        sytem_prompt = """You are a multilingual linguistics classifier agent.
            Given an input Reasoning and correct option, Your task is to determine whether the given REASONING is logically
and semantically aligned with the given QUESTION. Return a JSON object that strictly follows the provided JSON schema.
        
        Instructions:
            1. First give an one line summary of the REASONING identifying the problem without revealing the correct option. Summary should not contain any hint of the correct answer. Save the summary inside "summary" tag of the JSON.
            2. Identify the list of languages used in the REASONING other than English. Put it in the "lang_list" of the JSON.
            3. Determine whether the given REASONING is aligned with the QUESTION.
                Return True in the "lang_err" tag of the JSON:
                -- If it is logically and semantically aligned with the question.
                -- if the languages used in REASONING match with the languages in the QUESTION.
                Return False in the "lang_err" tag of the JSON:
                -- Answers a different question.
                -- Misinterprets key entities, conditions, or relationships. Logically and semantically not aligned.
                -- Uses medically irrelevant concepts due to language confusion. Uses languages other than the languages in the question.
            4. Determine repetition error:
                Return True in the "rept_err" tag of the JSON: 
                -- if the text contains unnatural or excessive repetition of words, subwords, or phrases indicating model degeneration.
                Return False in the "rept_err" tag of the JSON:
                -- if repetition is normal and not excessive.
            5. Determine reasoning error: 
                Return True in the "reason_err" tag of the JSON:
                -- if reasoning path is incorrect but semantically linked to the question.
                Return False in the "reason_err" tag of the JSON:
                 -- if reasoning path is correct and semantically linked to the question.
        
            Rules: 
            - Return ONLY valid JSON.
            - Do NOT add explanations or extra fields.
            - Use boolean values only for lang_err, rept and reason tags of the JSON.
            """
        critic_prompt= f"""Consider the Question, Answer Choices and the correct option, to analyze the given REASONING for summary, lang_list (list of languages), lang_err (language error), rept_err (repetation error) and reason_err (reasoning error): 
        Question: {gtl['eq']}
        Answer Choices: {gtl['eo']}
        correct option: {gtl['a']}
        REASONING : {generated_text}
        """
        lang_check.append({
                "role": "system",
                "content": [{"type": "text", "text": sytem_prompt }]
            })
        
        lang_check.append({
            "role": "user",
            "content": [{"type": "text", "text": critic_prompt }]
        })

        rformat = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "LangTest",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "summary": {
                                    "type": "string",
                                    "description": "Overall one-line assessment of language usage and reasoning alignment."
                                },
                
                                "lang_list": {
                                    "type": "array",
                                    "items": { "type": "string" },
                                    "description": "List of detected natural language names in the text"
                                },
                
                                "lang_err": {
                                    "type": "boolean",
                                    "description": "True if language misunderstanding affects meaning"
                                },
                                "rept_err": {
                                    "type": "boolean",
                                    "description": "True if content is meaninglessly repetitive"
                                },
                                "reason_err": {
                                    "type": "boolean",
                                    "description": "True if reasoning path is incorrect but semantically linked to the question"
                                }
                            },
                            "required": [
                                "summary",
                                "lang_list",
                                "lang_err",
                                "rept_err",
                                "reason_err",
                            ],
                            "additionalProperties": False
                        }
                    }
                }

        try:
            ########################
            api_key = self.openai_api_key
            api_base = self.openai_api_base
            model = self.openai_model
            temperature = self.openai_temperature
            
            self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
            self.openai_api_base = "https://api.openai.com/v1/chat/completions"
            self.openai_model = "gpt-4o-mini"
            self.openai_temperature = 0.5
            ##########################

            response = self._call_openai_api(
                messages=lang_check,
                max_tokens=100,
                rformat=rformat
            )

            generated_json = json.loads(response['choices'][0]['message']['content'])
            
            print(f"Critic Response= {generated_json}")
            return generated_json
        except Exception as e:
            logger.error(f"Exception in Critic open AI API call: {e}")
            return None
        finally:
            ########################            
            self.openai_api_key = api_key
            self.openai_api_base = api_base
            self.openai_model = model
            self.openai_temperature = temperature
            ##########################
                    
                

   
     
    def handle_ground_truth_rethink(
    self,
    generated_text: str,
    env_state: Dict[str, Any],
    conversation_history: List[Dict[str, Any]],
    gtl=None
) -> bool:
        """
        Ground-truth driven rethink with no-repeat-wrong-choices.

        1. if ans wrong 3 times-
            a. If the COT is not linked with the multilingual query
            b. if The COT has issues due to  repeation of same language tokens 
                Then rethink in English (code switch)
                
            c. If the COT from actor is wrong due to factual or logical error
                Then rethink in original language
        2. if ans wrong 4th time correct answer provided and asked to give logic 
        """
    
        pred = self.extract_boxed_answer(generated_text)
        if pred is None:
            pred="[Option could not be extracted due to formatting error]"
            print("Generated option not found. No answer Tag")
            #return False
            
            
        if self.rethink_count > self.max_rethinks:
            print("Maximum rethink reached.")
            return False
    
        self.last_boxed_answer = pred
    
        #gt = env_state.get("gt_answer") if env_state else None
        if not gtl:
            print("Ground Truth Not found.")
            return False
    
        gt = gtl['a']
    
        # Correct → accept
        if pred == gt:
            print("Ground Truth Matched")
            return False
        
        # Incorrect → record and rethink
        self.wrong_answers.add(pred)
        
        # # uncomment if Question needs to be sent again
        # q=env_state["initial_prompt"].replace("Analyze the medical images and answer the correct option from multiple-choice question",
        #                                     "Re-analyze the previously provided images and answer the correct option: \n")
        # q=q.replace("<image>","")


        
        wsum=f"Attempt : {self.rethink_count +1 } : Option - {pred} : incorrect reasoning."
        Lang_msg=""
        if not self.code_switch:
            # ---- (English re-ask) ----
            lang_check=self.handle_language_gpt(generated_text,conversation_history,gtl)
            
            if lang_check:
                wsum=f"Attempt : {self.rethink_count + 1} : Option - {pred} : {lang_check['summary']}"
                if lang_check['lang_err'] or lang_check['rept_err']:
                    print(f"Language error triggered | Language={lang_check['lang_err']}, repetition={lang_check['rept_err']}")
                    self.code_switch = True
                    Lang_msg =  f"""The previous response shows: {"logical and semantical language unalignment with the question" if lang_check['lang_err'] else ""} {" and " if lang_check['lang_err'] and lang_check['rept_err'] else ""} {"repetition of unnecessary tokens due to model degeneration." if lang_check['rept_err'] else "."}
    
                                Now consider the question translated in English:
                                
                                Question: {gtl['eq']} \nAnswer Choices: {gtl['eo']}
                                
                                Please reanalyze the previously provided images and, going forward, respond only in clear English.
                                Follow the instructions below: """
        
        self.wrong_answers.add(pred)
        self.wrong_sum.add(wsum)
        banned = ", ".join(sorted(self.wrong_answers))
        banned_sum=", ".join(sorted(self.wrong_sum))
        if self.rethink_count == self.max_rethinks:
            content =   f"""The correct option is {gt}.
                        Now, provide the reasoning for the correct answer in { "English" if self.code_switch else gtl['l']  }.
                        Use tools before answering if requird tools data is not availabe previously.
                        Provide answer within <answer> tag with a boxed option."""

                # # uncomment if Question needs to be sent again
                # f"{gtl['q']} Answer choices:{gtl['o']}."

        else:
            content= f"""<SHORT_MEMORY> Your current answer: {pred} is incorrect. Try again.</SHORT_MEMORY>
            <LONG_MEMORY> The Summary of your previous attempts: {banned_sum} The following options are incorrect from your previous attempts: {banned}. Exclude them from the given options :{ gtl['eo'] if self.code_switch else gtl['o'] }. Your answer must not be these options. </LONG_MEMORY>
            Analyse the previously given images very carefully to answer the question in { "English" if self.code_switch else gtl['l']  }
            --Use tools before answering if requird tools data is not availabe previously.
            --Do not answer with same option.
            --Provide answer within <answer> tag with a boxed option."""

            
        content = Lang_msg + content

        #print(f"handle_ground_truth_rethink: {content}")
        conversation_history.append({
        "role": "user",
        "content": [{"type": "text", "text": content }]
        })
    
        return True


#     def enforce_tool_usage_per_attempt(
#     self,
#     generated_text: str,
#     conversation_history: List[Dict[str, Any]]
# ) -> bool:
#         """
#         Enforce at least one tool call before answering.
    
#         Returns:
#             True  -> enforcement triggered
#             False -> OK to proceed
#         """
    
#         # No final answer yet → nothing to enforce
#         if "<answer>" not in generated_text.lower():
#             return False
    
#         # Tool already used → OK
#         if self.tool_used_in_attempt:
#             return False
    
        # Enforce tool usage
        # conversation_history.append({
        #     "role": "user",
        #     "content": (
        #         "You must call at least one tool before providing the final answer. "
        #         "Use the available tools to analyze the image, then answer."
        #     )
        # })
    
        # return True
################################################################
# Critic end
###############################################################


    def generate_conversation(self, system_prompt, initial_prompt, images=None,gtl=None):
        """
        Generate multi-turn conversation, handle tool calls and observations
        
        Args:
            system_prompt: System prompt text
            initial_prompt: Initial prompt text
            images: Initial image list (optional)
            env_state: Environment state (optional)
            
        Returns:
            conversation_history: Conversation history
        """
        conversation_history = []
        
        if system_prompt:
            conversation_history.append({
                "role": "system",
                "content": [{"type": "text", "text":system_prompt}]
            })
        
        initial_content = self._prepare_message_with_images(initial_prompt, images)

        
        
        conversation_history.append({
            "role": "user",
            "content": initial_content
        })
        
        #print(initial_content)
        
        current_mm_data = {}
        if images:
            current_mm_data["image"] = images
        current_mm_data["initial_prompt"] = initial_prompt
        
        self.rethink_count = 0
        self.wrong_answers.clear()
        self.wrong_sum.clear()
        
        while self.rethink_count <= (self.max_rethinks+1):
            # ---- Each rethink gets full max_turns ----
            print(f"Start trial: {self.rethink_count+1}")
            rethink_triggered = False
            
        
            for turn in range(self.max_turns):
                print(f"Start turn {turn+1}")
                
                try:
                    response = self._call_openai_api(
                        messages=conversation_history,
                        max_tokens=self.max_tokens_per_turn,
                        stop=["</tool_call>"]
                    )
                    
                    generated_text = response['choices'][0]['message']['content']
                    
                    if "<tool_call>" in generated_text and "</tool_call>" not in generated_text:
                        generated_text += "</tool_call>"
                   
                    ############################
                    if self.save_intermediate_responses:
                        print(f"Turn {turn+1} assistant reply::sample-{gtl.get('id', '')}::{generated_text}")
    
                    conversation_history.append({
                        "role": "assistant", 
                        "content": [{"type": "text", "text":generated_text}]
                    })
                    
                    has_action, all_actions_successful, observation = self.process_action_and_execute(
                        generated_text, 
                        current_mm_data
                    )
                    
                    # # Enforce tool usage
                    # if self.enforce_tool_usage_per_attempt(
                    #     generated_text,
                    #     conversation_history
                    # ):
                    #     continue


                    
                    # has_action=False #remove after test
                    # generated_text="""<think> रोगी के लक्षणों में लगातार बिगड़ते सिरदर्द, संज्ञानात्मक गिरावट, और हल्के फोकल तंत्रिका संबंधी घाटे शामिल हैं। MRI परिणामों में एक घाव है जो आसपास के मस्तिष्क ऊतकों की तुलना में अलग बनावट और सिग्नल तीव्रता दिखाता है। यह जानकारी यह सुझाव देती है कि घाव एक घातक या अज्ञात प्रकार का हो सकता है। विकल्पों का विश्लेषण करते हैं: - **मेनिन्जियोमा (Meningioma)**: आमतौर पर धीमी गति से बढ़ने वाला होता है, लेकिन यह आमतौर पर सिरदर्द का कारण बनता है। - **मेटास्टेटिक ट्यूमर (Metastatic Tumor)**: यह आमतौर पर एक घाव होता है जो अन्य अंगों से मस्तिष्क में फैलता है और त्वरित लक्षण उत्पन्न कर सकता है। - **डिफ्यूज़ ग्लिओमा (Diffuse Glioma)**: यह मस्तिष्क के ऊतकों में फैलता है और आमतौर पर अधिक आक्रामक होता है। - **प्राइमरी ब्रेन लिंफोमा (Primary Brain Lymphoma)**: यह एक दुर्लभ लेकिन आक्रामक ट्यूमर है जो आमतौर আমি চিত্র B-এর বিশ্লেষণের জন্য টুল ব্যবহার করতে চাইছিলাম, কিন্তু চিত্রটি উপলব্ধ নয়। এর ফলে, আমি সরাসরি টুল ব্যবহার করতে পারছি না। যেহেতু প্রশ্নটি নিডল পাংচারের পর কী পরিবর্তন ঘটে তা নিয়ে,पर इम्युनोकोम्प्रोमाइज्ड रोगियों में होता है। - **सबड्यूरल हेमेटोमा (Subdural Hematoma)**: यह आमतौर पर एक आघात के बाद होता है और लक्षण जल्दी विकसित होते हैं। रोगी के लक्षणों और MRI निष्कर्षों को देखते हुए, मेटास्टेटिक ट्यूमर सबसे संभावित निदान हो सकता है क्योंकि यह आमतौर पर त्वरित लक्षण उत्पन्न करता है और अलग बनावट और सिग्नल तीव्रता दिखा सकता है। </think> <answer> \boxed{B} </answer>""" #remove after test
                    if not has_action: 
                        if gtl:
                            #print("GT TRUE")
                            if self.handle_ground_truth_rethink(
                                generated_text,
                                current_mm_data,
                                conversation_history,
                                gtl
                            ):
                                rethink_triggered = True
                                self.rethink_count += 1
                                break   # restart attempt (turn resets)
                        else:
                            print("Ground Truth NOT provided. handle_ground_truth_rethink not called.")

                    if not has_action or turn == self.max_turns - 1:
                        break
                    
                    obs_text = observation.get("text", "")
                    obs_images = observation.get("image", None)
                    
                    if self.save_intermediate_responses:
                        #print(f"Turn {turn+1} observation: {obs_text}")
                        if not all_actions_successful:
                            print(f"Warning: Turn {turn+1} tool call failed")
                    
                    obs_message_content = self._prepare_message_with_images(obs_text, obs_images)
                    
                    conversation_history.append({
                        "role": "user",
                        "content": obs_message_content
                    })
                    
                    if obs_images is not None:
                        if current_mm_data is None:
                            current_mm_data = {"image": obs_images}
                        elif "image" not in current_mm_data:
                            current_mm_data["image"] = obs_images
                        else:
                            current_mm_data["image"].extend(obs_images)
                    
                except Exception as e:
                    logger.error(f"Exception in turn {turn+1}: {str(e)}")
                    break
                
                print(f"Turn {turn+1} finished")
            if rethink_triggered:
                continue  # NEW attempt, turn counter resets   
                
            # Inner loop finished without rethink → stop
            return conversation_history

        # ---- We only reach here if rethink triggered ----
        # Continue outer while → fresh max_turns
        return conversation_history

    def chat_with_tools(self, prompt, images=None, system_prompt=None,gtl=None):
        """
        Simplified API entry for multi-turn tool interaction conversation
        
        Args:
            prompt: User question
            images: Optional image list
            system_prompt: Optional system prompt
            env_state: Optional environment state
            
        Returns:
            final_response: Final response text
            conversation: Full conversation history
        """
        
        images_copy = deepcopy(images) if images else None
        
        conversation = self.generate_conversation(system_prompt, prompt, images_copy,gtl)
        
        final_response = None
        for message in reversed(conversation):
            if message["role"] == "assistant":
                final_response = message["content"]
                break
                
        return final_response, conversation