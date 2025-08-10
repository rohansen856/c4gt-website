import json
import logging
from flask import render_template, request, jsonify, g
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import torch

class SEALionHandler:
    
    model = None
    tokenizer = None
    model_loaded = False
    
    @classmethod
    def initialize_model(cls):
        if cls.model_loaded:
            return True
            
        try:
            model_name = "aisingapore/llama-sea-lion-v3.5-8b-r"
            
            cls.tokenizer = AutoTokenizer.from_pretrained(model_name)
            cls.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            cls.model_loaded = True
            logging.info("SEALion model loaded successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load SEALion model: {str(e)}")
            return False
    
    @classmethod
    def generate_text(cls, prompt, max_length=512, temperature=0.7, top_p=0.9):
        if not cls.model_loaded:
            if not cls.initialize_model():
                return None
        
        try:
            inputs = cls.tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = cls.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=cls.tokenizer.eos_token_id
                )
            
            generated_text = cls.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_text = generated_text[len(prompt):].strip()
            
            return response_text
            
        except Exception as e:
            logging.error(f"Error generating text: {str(e)}")
            return None
    
    @classmethod
    def get(cls):
        user = g.handler.get_current_user()
        if not user:
            return render_template('userlogin.html', message="Please log in to use SEALion features")
        
        return render_template('sealion.html', user=user)
    
    @classmethod
    def post(cls):
        user = g.handler.get_current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        try:
            data = request.get_json()
            prompt = data.get('prompt', '').strip()
            
            if not prompt:
                return jsonify({"error": "Prompt is required"}), 400
            
            max_length = data.get('max_length', 512)
            temperature = data.get('temperature', 0.7)
            top_p = data.get('top_p', 0.9)
            
            generated_text = cls.generate_text(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            
            if generated_text is None:
                return jsonify({"error": "Failed to generate text. Model may not be loaded."}), 500
            
            return jsonify({
                "generated_text": generated_text,
                "prompt": prompt,
                "model": "SEALion v3.5 8B"
            })
            
        except Exception as e:
            logging.error(f"Error in SEALionHandler.post: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @classmethod
    def translate_post(cls):
        user = g.handler.get_current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        try:
            data = request.get_json()
            text = data.get('text', '').strip()
            target_language = data.get('target_language', 'English')
            
            if not text:
                return jsonify({"error": "Text is required"}), 400
            
            prompt = f"Translate the following text to {target_language}:\n\n{text}\n\nTranslation:"
            
            translated_text = cls.generate_text(
                prompt=prompt,
                max_length=1024,
                temperature=0.3,
                top_p=0.8
            )
            
            if translated_text is None:
                return jsonify({"error": "Failed to translate text"}), 500
            
            return jsonify({
                "translated_text": translated_text.strip(),
                "original_text": text,
                "target_language": target_language,
                "model": "SEALion v3.5 8B"
            })
            
        except Exception as e:
            logging.error(f"Error in SEALionHandler.translate_post: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
    
    @classmethod
    def summarize_post(cls):
        user = g.handler.get_current_user()
        if not user:
            return jsonify({"error": "Authentication required"}), 401
        
        try:
            data = request.get_json()
            text = data.get('text', '').strip()
            
            if not text:
                return jsonify({"error": "Text is required"}), 400
            
            prompt = f"Please provide a concise summary of the following text:\n\n{text}\n\nSummary:"
            
            summary = cls.generate_text(
                prompt=prompt,
                max_length=1024,
                temperature=0.5,
                top_p=0.8
            )
            
            if summary is None:
                return jsonify({"error": "Failed to generate summary"}), 500
            
            return jsonify({
                "summary": summary.strip(),
                "original_text": text,
                "model": "SEALion v3.5 8B"
            })
            
        except Exception as e:
            logging.error(f"Error in SEALionHandler.summarize_post: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500