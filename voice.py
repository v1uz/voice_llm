#!/usr/bin/env python3
"""
ULTIMATE Voice Assistant with AI Agent Capabilities
- Multiple personalities
- System command execution
- Smart action detection
- Performance optimized
- Python 3.13 compatible
"""

import torch
import pyaudio
import wave
import ollama
import time
import os
import sys
import asyncio
import edge_tts
import pygame
import threading
import logging
import subprocess
import webbrowser
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import speech_recognition as sr
import numpy as np
import tempfile
import warnings
import random
import atexit
import re
from urllib.parse import quote_plus

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Centralized configuration"""
    # Audio settings
    SAMPLE_RATE: int = 16000
    CHUNK_SIZE: int = 1024
    CHANNELS: int = 1
    AUDIO_FORMAT = pyaudio.paInt16
    
    # VAD settings
    SILENCE_THRESHOLD: float = 2.0
    ENERGY_THRESHOLD: float = 100
    MAX_RECORDING_DURATION: int = 40
    
    # Model settings
    WHISPER_MODEL: str = "tiny"
    WHISPER_BEAM_SIZE: int = 2
    OLLAMA_MODEL: str = "llama3.2"
    
    # TTS settings
    TTS_VOICE: str = "en-US-ChristopherNeural"
    TTS_RATE: str = "+30%"
    
    # Conversation settings
    MAX_HISTORY_LENGTH: int = 8
    RESPONSE_LENGTHS: Dict[str, int] = field(default_factory=lambda: {
        "short": 100,
        "medium": 250,
        "long": 500,
        "essay": 2048
    })
    
    # Action system settings
    ENABLE_ACTIONS: bool = True
    REQUIRE_CONFIRMATION: bool = True
    DANGEROUS_COMMANDS: List[str] = field(default_factory=lambda: [
        'rm', 'del', 'format', 'shutdown', 'reboot', 'kill', 'pkill'
    ])

CONFIG = Config()

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL STATE
# ============================================================================

interrupt_flag = threading.Event()
pygame.mixer.init()
whisper_model = None
action_log = []

# ============================================================================
# CLEANUP HANDLER
# ============================================================================

def cleanup_temp_files():
    """Clean up temporary audio files on exit"""
    for pattern in ["temp_*.wav", "temp_*.mp3"]:
        for temp_file in Path(".").glob(pattern):
            try:
                temp_file.unlink()
                logger.debug(f"Cleaned up {temp_file}")
            except OSError as e:
                logger.warning(f"Could not remove {temp_file}: {e}")

atexit.register(cleanup_temp_files)

# ============================================================================
# PERSONALITY SYSTEM
# ============================================================================

class PersonalityManager:
    """Manage multiple AI personalities"""
    
    PERSONALITIES = {
        "bro": {
            "name": "Bro Mode",
            "prompt": """You are my bro, my homie, my dawg. You speak like a chill 18-year-old who's into coding and tech but keeps it real. 
You use slang, you're funny, and you're not afraid to roast me. Keep responses SHORT and conversational - we're having a voice chat, not writing essays.
Sound natural like we're just chillin. Use 'bro', 'dawg', 'fam', 'bet', 'no cap', 'fr fr' and other Gen Z slang.
If I ask for code, you got me, but explain it like we're homies, not in a classroom.
When I ask you to perform actions like opening websites or files, you can do it and respond naturally.""",
            "voice": "en-US-ChristopherNeural"
        },
        
        "teacher": {
            "name": "Teacher Mode",
            "prompt": """You are a patient, knowledgeable teacher who loves helping students learn. You explain concepts clearly with examples.
You're encouraging and break down complex topics into simple steps. You ask questions to check understanding.
Keep responses conversational since this is voice chat. Use metaphors and real-world examples.
When performing actions, explain what you're doing and why.""",
            "voice": "en-US-GuyNeural"
        },
        
        "professional": {
            "name": "Professional Mode",
            "prompt": """You are a professional AI assistant. You communicate clearly, concisely, and formally.
You focus on efficiency and accuracy. Your tone is respectful and business-appropriate.
Keep responses brief and to the point since this is voice communication.
When executing actions, confirm completion professionally.""",
            "voice": "en-US-GuyNeural"
        },
        
        "hacker": {
            "name": "Hacker Mode",
            "prompt": """You are a cyberpunk AI with elite hacking skills. You speak in technical terms mixed with hacker slang.
Use terms like 'pwned', 'root access', 'exploit', '1337', 'zero-day'. You're confident and slightly edgy.
Keep responses concise for voice chat. Explain technical concepts in hacker style.
When performing actions, describe them like you're executing a hack.""",
            "voice": "en-US-ChristopherNeural"
        },
        
        "assistant": {
            "name": "Default Assistant",
            "prompt": """You are a helpful, friendly AI assistant. You're conversational and warm without being overly casual.
You provide clear, accurate information and can perform various tasks.
Keep responses appropriate for voice chat - concise but complete.
When executing actions, confirm them clearly.""",
            "voice": "en-US-JennyNeural"
        }
    }
    
    def __init__(self, default_personality="bro"):
        self.current_personality = default_personality
        logger.info(f"Personality initialized: {self.PERSONALITIES[default_personality]['name']}")
    
    def set_personality(self, personality: str) -> bool:
        """Switch to a different personality"""
        if personality in self.PERSONALITIES:
            self.current_personality = personality
            logger.info(f"Personality changed to: {self.PERSONALITIES[personality]['name']}")
            return True
        return False
    
    def get_system_prompt(self) -> str:
        """Get current personality's system prompt"""
        return self.PERSONALITIES[self.current_personality]["prompt"]
    
    def get_voice(self) -> str:
        """Get current personality's preferred voice"""
        return self.PERSONALITIES[self.current_personality]["voice"]
    
    def list_personalities(self) -> List[str]:
        """Get list of available personalities"""
        return list(self.PERSONALITIES.keys())
    
    def get_personality_info(self, personality: str) -> Optional[Dict]:
        """Get info about a personality"""
        return self.PERSONALITIES.get(personality)

personality_manager = PersonalityManager()

# ============================================================================
# ACTION SYSTEM
# ============================================================================

class ActionExecutor:
    """Execute system actions safely"""
    
    def __init__(self):
        self.enabled = CONFIG.ENABLE_ACTIONS
        self.require_confirmation = CONFIG.REQUIRE_CONFIRMATION
    
    def is_dangerous(self, command: str) -> bool:
        """Check if command is potentially dangerous"""
        command_lower = command.lower()
        return any(dangerous in command_lower for dangerous in CONFIG.DANGEROUS_COMMANDS)
    
    def get_confirmation(self, action: str) -> bool:
        """Ask user for confirmation"""
        if not self.require_confirmation:
            return True
        
        print(f"\n⚠️  Confirm action: {action}")
        response = input("Execute? (yes/no): ").strip().lower()
        return response in ['yes', 'y']
    
    def open_website(self, url: str) -> Tuple[bool, str]:
        """Open a website in default browser"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            if self.is_dangerous(url) or not self.get_confirmation(f"Open website: {url}"):
                return False, "Action cancelled by user"
            
            webbrowser.open(url)
            action_log.append({"action": "open_website", "url": url, "time": time.time()})
            return True, f"Opened {url}"
        except Exception as e:
            logger.error(f"Failed to open website: {e}")
            return False, str(e)
    
    def google_search(self, query: str) -> Tuple[bool, str]:
        """Perform a Google search"""
        try:
            search_url = f"https://www.google.com/search?q={quote_plus(query)}"
            
            if not self.get_confirmation(f"Google search: {query}"):
                return False, "Action cancelled by user"
            
            webbrowser.open(search_url)
            action_log.append({"action": "google_search", "query": query, "time": time.time()})
            return True, f"Searched Google for: {query}"
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            return False, str(e)
    
    def open_file(self, filepath: str) -> Tuple[bool, str]:
        """Open a file with default application"""
        try:
            path = Path(filepath).expanduser()
            
            if not path.exists():
                return False, f"File not found: {filepath}"
            
            if not self.get_confirmation(f"Open file: {filepath}"):
                return False, "Action cancelled by user"
            
            if sys.platform == 'win32':
                os.startfile(path)
            elif sys.platform == 'darwin':
                subprocess.run(['open', path])
            else:
                subprocess.run(['xdg-open', path])
            
            action_log.append({"action": "open_file", "path": str(path), "time": time.time()})
            return True, f"Opened {path.name}"
        except Exception as e:
            logger.error(f"Failed to open file: {e}")
            return False, str(e)
    
    def open_application(self, app_name: str) -> Tuple[bool, str]:
        """Launch an application"""
        try:
            if not self.get_confirmation(f"Launch application: {app_name}"):
                return False, "Action cancelled by user"
            
            if sys.platform == 'win32':
                subprocess.Popen([app_name], shell=True)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', '-a', app_name])
            else:
                subprocess.Popen([app_name])
            
            action_log.append({"action": "open_app", "app": app_name, "time": time.time()})
            return True, f"Launched {app_name}"
        except Exception as e:
            logger.error(f"Failed to launch app: {e}")
            return False, str(e)
    
    def execute_command(self, command: str) -> Tuple[bool, str]:
        """Execute a shell command (with safety checks)"""
        try:
            if self.is_dangerous(command):
                return False, "Command blocked: potentially dangerous"
            
            if not self.get_confirmation(f"Execute command: {command}"):
                return False, "Action cancelled by user"
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            action_log.append({"action": "execute_command", "command": command, "time": time.time()})
            
            if result.returncode == 0:
                return True, result.stdout or "Command executed successfully"
            else:
                return False, result.stderr or "Command failed"
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            logger.error(f"Failed to execute command: {e}")
            return False, str(e)

action_executor = ActionExecutor()

# ============================================================================
# INTENT DETECTION
# ============================================================================

class IntentDetector:
    """Detect user intent and extract action parameters"""
    
    # Action patterns
    PATTERNS = {
        'open_website': [
            r'open\s+(?:website\s+)?(.+\.(?:com|org|net|edu|gov|io))',
            r'go\s+to\s+(.+\.(?:com|org|net|edu|gov|io))',
            r'browse\s+(?:to\s+)?(.+\.(?:com|org|net|edu|gov|io))',
        ],
        'google_search': [
            r'(?:google|search)\s+(?:for\s+)?(.+)',
            r'look\s+up\s+(.+)',
            r'find\s+(?:me\s+)?(?:information\s+(?:about|on)\s+)?(.+)',
        ],
        'open_file': [
            r'open\s+(?:file\s+)?(.+)',
            r'launch\s+(?:file\s+)?(.+)',
        ],
        'open_app': [
            r'open\s+(?:application|app|program)\s+(.+)',
            r'launch\s+(.+)',
            r'start\s+(.+)',
        ],
    }
    
    def detect_intent(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect intent and extract parameters
        Returns: (intent_type, parameter)
        """
        text_lower = text.lower().strip()
        
        # Check each pattern
        for intent, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    param = match.group(1).strip()
                    logger.info(f"Intent detected: {intent} -> {param}")
                    return intent, param
        
        return None, None
    
    def should_execute(self, text: str) -> bool:
        """Quick check if text contains action keywords"""
        action_keywords = ['open', 'search', 'google', 'launch', 'start', 'browse', 'find', 'look up']
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in action_keywords)

intent_detector = IntentDetector()

# ============================================================================
# TEXT-TO-SPEECH
# ============================================================================

async def speak_async(text: str, voice: Optional[str] = None):
    """Fast async TTS with interrupt checking"""
    if voice is None:
        voice = personality_manager.get_voice()
    
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_mp3 = f.name
        
        communicate = edge_tts.Communicate(text, voice, rate=CONFIG.TTS_RATE)
        await communicate.save(temp_mp3)
        
        if interrupt_flag.is_set():
            logger.info("Speech generation interrupted")
            try:
                os.remove(temp_mp3)
            except:
                pass
            return
        
        try:
            pygame.mixer.music.load(temp_mp3)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                if interrupt_flag.is_set():
                    pygame.mixer.music.stop()
                    logger.info("Playback interrupted")
                    break
                pygame.time.wait(10)
        
        finally:
            pygame.mixer.music.unload()
            try:
                os.remove(temp_mp3)
            except OSError as e:
                logger.warning(f"Could not remove temp file: {e}")
            
    except Exception as e:
        logger.error(f"TTS Error: {e}", exc_info=True)

def speak(text: str) -> threading.Thread:
    """Wrapper for non-blocking speech"""
    logger.info(f"🤖 Assistant: {text}")
    interrupt_flag.clear()
    
    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(speak_async(text))
        finally:
            loop.close()
    
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread

# ============================================================================
# BACKGROUND INTERRUPT LISTENER
# ============================================================================

def start_background_listener() -> callable:
    """Listen for stop commands while AI is speaking"""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = CONFIG.ENERGY_THRESHOLD
    recognizer.dynamic_energy_threshold = True
    
    try:
        mic = sr.Microphone()
    except OSError as e:
        logger.error(f"Could not access microphone: {e}")
        return None
    
    def callback(recognizer, audio):
        try:
            text = recognizer.recognize_google(audio, language="en-US").lower()
            logger.debug(f"Background heard: '{text}'")
            
            stop_words = ['stop', 'shut up', 'quiet', 'pause', 'silence']
            if any(word in text for word in stop_words):
                interrupt_flag.set()
                logger.info("Stop command detected!")
                
        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            logger.warning(f"Google API error: {e}")
        except Exception as e:
            logger.error(f"Background listener error: {e}")
    
    try:
        stop_listening = recognizer.listen_in_background(
            mic, 
            callback, 
            phrase_time_limit=2
        )
        logger.info("Background listener started")
        return stop_listening
    except Exception as e:
        logger.error(f"Could not start background listener: {e}")
        return None

# ============================================================================
# AUDIO RECORDING WITH VAD
# ============================================================================

def record_audio_vad(
    max_duration: int = None,
    silence_threshold: float = None,
    energy_threshold: float = None
) -> Optional[str]:
    """Record audio with voice activity detection"""
    if max_duration is None:
        max_duration = CONFIG.MAX_RECORDING_DURATION
    if silence_threshold is None:
        silence_threshold = CONFIG.SILENCE_THRESHOLD
    if energy_threshold is None:
        energy_threshold = CONFIG.ENERGY_THRESHOLD
    
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(
            format=CONFIG.AUDIO_FORMAT,
            channels=CONFIG.CHANNELS,
            rate=CONFIG.SAMPLE_RATE,
            input=True,
            frames_per_buffer=CONFIG.CHUNK_SIZE
        )
    except OSError as e:
        logger.error(f"Could not open audio stream: {e}")
        p.terminate()
        return None
    
    logger.info("🎤 Listening...")
    
    frames = []
    silence_chunks = 0
    silence_threshold_chunks = int(silence_threshold * CONFIG.SAMPLE_RATE / CONFIG.CHUNK_SIZE)
    has_spoken = False
    max_energy = 0
    
    max_iterations = int(max_duration * CONFIG.SAMPLE_RATE / CONFIG.CHUNK_SIZE)
    for i in range(max_iterations):
        try:
            data = stream.read(CONFIG.CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)
            
            audio_data = np.frombuffer(data, dtype=np.int16)
            if len(audio_data) > 0:
                energy = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
                max_energy = max(max_energy, energy)
                
                if energy > energy_threshold:
                    silence_chunks = 0
                    if not has_spoken:
                        has_spoken = True
                        logger.info("🗣️  Speech detected...")
                else:
                    if has_spoken:
                        silence_chunks += 1
                        
                if has_spoken and silence_chunks >= silence_threshold_chunks:
                    logger.info("✓ Recording complete")
                    break
                    
        except Exception as e:
            logger.error(f"Audio read error: {e}")
            break
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    if not frames:
        logger.warning("No audio frames recorded")
        return None
        
    if not has_spoken:
        logger.warning(f"No speech detected (max energy: {max_energy:.1f})")
        logger.info("Try speaking louder or adjusting energy threshold")
        return None
    
    temp_file = "temp_audio.wav"
    try:
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(CONFIG.CHANNELS)
            wf.setsampwidth(p.get_sample_size(CONFIG.AUDIO_FORMAT))
            wf.setframerate(CONFIG.SAMPLE_RATE)
            wf.writeframes(b''.join(frames))
        return temp_file
    except Exception as e:
        logger.error(f"Could not save audio file: {e}")
        return None

# ============================================================================
# WHISPER INITIALIZATION
# ============================================================================

def initialize_whisper():
    """Initialize Whisper model"""
    global whisper_model
    
    logger.info("Loading Faster Whisper...")
    
    try:
        from faster_whisper import WhisperModel
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        
        whisper_model = WhisperModel(
            CONFIG.WHISPER_MODEL,
            device=device,
            compute_type=compute_type,
            cpu_threads=4,
            num_workers=2
        )
        
        logger.info(f"✓ Whisper loaded ({device.upper()})")
        return True
        
    except ImportError:
        logger.error("faster-whisper not installed! Run: pip install faster-whisper")
        return False
    except Exception as e:
        logger.error(f"Could not load Whisper: {e}")
        return False

# ============================================================================
# TRANSCRIPTION
# ============================================================================

def transcribe_audio(audio_filepath: str) -> Optional[str]:
    """Transcribe audio file to text"""
    if not whisper_model:
        logger.error("Whisper model not initialized")
        return None
    
    start_time = time.time()
    
    try:
        segments, info = whisper_model.transcribe(
            audio_filepath,
            beam_size=CONFIG.WHISPER_BEAM_SIZE,
            language="en",
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                threshold=0.5
            ),
            word_timestamps=False,
            temperature=0
        )
        
        text = " ".join([segment.text.strip() for segment in segments])
        
        duration = time.time() - start_time
        logger.info(f"⚡ Transcribed in {duration:.2f}s")
        
        return text.strip()
        
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return None

# ============================================================================
# AI CHAT WITH ACTION SUPPORT
# ============================================================================

def chat_with_ai(
    prompt: str, 
    conversation_history: List[Dict],
    response_length: str = "medium"
) -> str:
    """Chat with AI assistant with action support"""
    
    system_prompt = personality_manager.get_system_prompt()
    
    messages = [{'role': 'system', 'content': system_prompt}]
    messages.extend(conversation_history[-CONFIG.MAX_HISTORY_LENGTH:])
    messages.append({'role': 'user', 'content': prompt})
    
    num_predict = CONFIG.RESPONSE_LENGTHS.get(response_length, 250)
    
    try:
        response = ollama.chat(
            model=CONFIG.OLLAMA_MODEL,
            messages=messages,
            options={
                'temperature': 0.8,
                'num_predict': num_predict,
                'top_p': 0.9,
                'repeat_penalty': 1.1,
            }
        )
        return response['message']['content']
    except Exception as e:
        logger.error(f"AI chat error: {e}", exc_info=True)
        return "Yo my bad, something went wrong. Try again?"

# ============================================================================
# PROCESS USER INPUT WITH ACTIONS
# ============================================================================

def process_input_with_actions(text: str, conversation_history: List[Dict]) -> str:
    """Process user input and execute actions if needed"""
    
    # Check if this might be an action request
    if intent_detector.should_execute(text):
        intent, param = intent_detector.detect_intent(text)
        
        if intent and param:
            logger.info(f"Executing action: {intent}({param})")
            
            # Execute the action
            success = False
            message = ""
            
            if intent == 'open_website':
                success, message = action_executor.open_website(param)
            elif intent == 'google_search':
                success, message = action_executor.google_search(param)
            elif intent == 'open_file':
                success, message = action_executor.open_file(param)
            elif intent == 'open_app':
                success, message = action_executor.open_application(param)
            
            # Generate AI response about the action
            if success:
                action_context = f"I just executed this action: {message}. Respond naturally about completing this task."
            else:
                action_context = f"I tried to execute an action but failed: {message}. Explain this to the user."
            
            # Get AI to respond about the action
            conversation_history.append({'role': 'system', 'content': action_context})
            response = chat_with_ai(text, conversation_history, response_length="short")
            conversation_history.pop()  # Remove action context
            
            return response
    
    # Normal conversation
    return chat_with_ai(text, conversation_history)

# ============================================================================
# MAIN ASSISTANT
# ============================================================================

def run_voice_assistant():
    """Main voice assistant loop"""
    print("\n" + "="*70)
    print("🤖 ULTIMATE AI VOICE ASSISTANT - Agent Mode Enabled")
    print("="*70)
    print(f"\n💡 Current Personality: {personality_manager.PERSONALITIES[personality_manager.current_personality]['name']}")
    print("\n🎯 Quick Tips:")
    print("  • Press ENTER to start talking")
    print("  • Say 'open google.com' or 'search for Python tutorials'")
    print("  • Say 'stop' while I'm talking to interrupt me")
    print("  • Type '/help' for all commands")
    print("="*70)
    
    conversation_history = []
    
    # Start background listener
    stop_listener = start_background_listener()
    
    # Greeting based on personality
    greetings = {
        "bro": [
            "Yo what's good! Your AI homie with superpowers is here. I can browse, search, open stuff - whatever you need fam!",
            "Ayy we lit! Press Enter and let's get it. I got agent mode enabled so I can actually do stuff now, no cap!",
        ],
        "teacher": [
            "Hello! I'm ready to help you learn and explore. I can also assist with opening files, searching, and more!",
        ],
        "professional": [
            "Good day. AI assistant ready. I can execute commands, search information, and manage files efficiently.",
        ],
        "hacker": [
            "System initialized. Agent protocols active. Ready to execute your commands, root access granted.",
        ],
        "assistant": [
            "Hello! I'm your AI assistant. I can help with information, tasks, and system actions. How can I help?",
        ]
    }
    
    greeting_list = greetings.get(personality_manager.current_personality, greetings["assistant"])
    speak_thread = speak(random.choice(greeting_list))
    speak_thread.join()
    
    while True:
        try:
            interrupt_flag.clear()
            user_input = input("\n> ").strip()
            
            # Handle exit
            if user_input.lower() in ['quit', 'exit', 'bye', 'peace', 'q']:
                farewells = {
                    "bro": ["Aight bet, catch you later bro!", "Peace out homie!", "Later fam!"],
                    "teacher": ["Goodbye! Keep learning!", "Have a great day!"],
                    "professional": ["Goodbye. Session terminated."],
                    "hacker": ["Disconnecting. Stay elite."],
                    "assistant": ["Goodbye! Have a great day!"]
                }
                farewell_list = farewells.get(personality_manager.current_personality, farewells["assistant"])
                speak(random.choice(farewell_list))
                time.sleep(2)
                break
            
            # Handle commands
            elif user_input.startswith('/'):
                if user_input == '/clear':
                    conversation_history.clear()
                    speak("History cleared!")
                
                elif user_input == '/persona':
                    print("\n📋 Available personalities:")
                    for key, value in personality_manager.PERSONALITIES.items():
                        print(f"  • {key}: {value['name']}")
                    print("\nChange with: /persona <name>")
                
                elif user_input.startswith('/persona '):
                    persona_name = user_input.split(' ', 1)[1].lower()
                    if personality_manager.set_personality(persona_name):
                        info = personality_manager.get_personality_info(persona_name)
                        speak(f"Personality switched to {info['name']}!")
                    else:
                        print(f"Unknown personality: {persona_name}")
                
                elif user_input == '/actions':
                    print("\n🎬 Recent actions:")
                    if action_log:
                        for i, action in enumerate(action_log[-10:], 1):
                            print(f"  {i}. {action['action']}: {action.get('url', action.get('query', action.get('path', action.get('app', action.get('command', 'N/A')))))}")
                    else:
                        print("  No actions executed yet")
                
                elif user_input == '/history':
                    print(f"\n💬 Conversation history ({len(conversation_history)} messages):")
                    for i, msg in enumerate(conversation_history[-10:], 1):
                        role = "You" if msg['role'] == 'user' else "AI"
                        content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                        print(f"  {i}. {role}: {content}")
                
                elif user_input == '/help':
                    print("\n📖 Available commands:")
                    print("  /clear        - Clear conversation history")
                    print("  /persona      - Show available personalities")
                    print("  /persona <n>  - Switch personality")
                    print("  /actions      - Show recent actions")
                    print("  /history      - Show conversation history")
                    print("  /help         - Show this help")
                    print("  quit/q        - Exit assistant")
                    print("\n🎬 Action examples:")
                    print("  'open google.com'")
                    print("  'search for Python tutorials'")
                    print("  'open notepad'")
                
                else:
                    print("Unknown command. Type '/help' for available commands.")
                
                continue
            
            # Voice input
            elif user_input == '':
                audio_file = record_audio_vad()
                
                if audio_file and os.path.exists(audio_file):
                    text = transcribe_audio(audio_file)
                    
                    if text:
                        print(f"\n💬 You: {text}")
                        
                        # Process with action detection
                        response = process_input_with_actions(text, conversation_history)
                        
                        # Update history
                        conversation_history.append({'role': 'user', 'content': text})
                        conversation_history.append({'role': 'assistant', 'content': response})
                        
                        # Speak
                        interrupt_flag.clear()
                        speak_thread = speak(response)
                        speak_thread.join()
                    
                    try:
                        os.remove(audio_file)
                    except OSError:
                        pass
            
            # Text input
            else:
                # Process with action detection
                response = process_input_with_actions(user_input, conversation_history)
                
                conversation_history.append({'role': 'user', 'content': user_input})
                conversation_history.append({'role': 'assistant', 'content': response})
                
                speak_thread = speak(response)
                speak_thread.join()
                
        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted!")
            interrupt_flag.set()
            speak("Goodbye!")
            break
        
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            print(f"❌ Error: {e}")
    
    # Cleanup
    if stop_listener:
        try:
            stop_listener(wait_for_stop=False)
        except:
            pass

# ============================================================================
# INITIALIZATION
# ============================================================================

def check_dependencies() -> bool:
    """Check all dependencies"""
    all_ok = True
    
    # Check Ollama
    try:
        ollama.list()
        logger.info("✓ Ollama is running")
    except Exception as e:
        logger.error(f"❌ Ollama not running: {e}")
        all_ok = False
    
    # Check Ollama model
    try:
        ollama.chat(
            model=CONFIG.OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': 'test'}]
        )
        logger.info(f"✓ {CONFIG.OLLAMA_MODEL} model available")
    except Exception as e:
        logger.warning(f"Model {CONFIG.OLLAMA_MODEL} not found")
        logger.info(f"Downloading {CONFIG.OLLAMA_MODEL}...")
        try:
            import subprocess
            subprocess.run(['ollama', 'pull', CONFIG.OLLAMA_MODEL], check=True)
            logger.info("✓ Model downloaded")
        except Exception as e:
            logger.error(f"❌ Could not download model: {e}")
            all_ok = False
    
    # Check PyAudio
    try:
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        p.terminate()
        logger.info(f"✓ PyAudio working ({device_count} devices)")
    except Exception as e:
        logger.error(f"❌ PyAudio error: {e}")
        all_ok = False
    
    # Check Whisper
    if not initialize_whisper():
        all_ok = False
    
    return all_ok

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("🤖 ULTIMATE AI ASSISTANT - Agent Mode")
    print("="*70 + "\n")
    
    logger.info("Checking dependencies...")
    if not check_dependencies():
        logger.error("❌ Dependency check failed!")
        print("\nFix errors above and try again.")
        sys.exit(1)
    
    cleanup_temp_files()
    
    print("\n" + "="*70)
    print("MAIN MENU")
    print("="*70)
    print("1. Start Voice Assistant")
    print("2. Exit")
    
    choice = input("\nYour choice (1-2): ").strip()
    
    if choice == "1":
        run_voice_assistant()
    else:
        print("Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()
