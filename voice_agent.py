#!/usr/bin/env python3
"""
AI Voice Agent - Autonomous voice assistant with planning and tools
Combines voice interface with autonomous agent capabilities
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Import agent system
from agent import AIAgent

# Import voice components from original voice.py
import torch
import pyaudio
import ollama
import edge_tts
import pygame
import threading
import speech_recognition as sr
import numpy as np
import tempfile
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# VOICE INTERFACE (from original voice.py)
# ============================================================================

class VoiceInterface:
    """Handles voice input/output"""

    def __init__(self, sample_rate=16000, energy_threshold=100):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.whisper_model = None
        self.interrupt_flag = threading.Event()

        pygame.mixer.init()

        # Initialize Whisper
        self._init_whisper()

    def _init_whisper(self):
        """Initialize Whisper model"""
        logger.info("Loading Faster Whisper...")

        try:
            from faster_whisper import WhisperModel

            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if torch.cuda.is_available() else "int8"

            self.whisper_model = WhisperModel(
                "tiny",
                device=device,
                compute_type=compute_type,
                cpu_threads=4,
                num_workers=2
            )

            logger.info(f"✓ Whisper loaded ({device.upper()})")

        except Exception as e:
            logger.error(f"Could not load Whisper: {e}")
            sys.exit(1)

    def record_audio(self, max_duration=40, silence_threshold=2.0):
        """Record audio with voice activity detection"""
        p = pyaudio.PyAudio()

        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=1024
            )
        except OSError as e:
            logger.error(f"Could not open audio stream: {e}")
            p.terminate()
            return None

        logger.info("🎤 Listening...")

        frames = []
        silence_chunks = 0
        silence_threshold_chunks = int(silence_threshold * self.sample_rate / 1024)
        has_spoken = False

        max_iterations = int(max_duration * self.sample_rate / 1024)

        for i in range(max_iterations):
            try:
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(data)

                audio_data = np.frombuffer(data, dtype=np.int16)
                if len(audio_data) > 0:
                    energy = np.sqrt(np.mean(audio_data.astype(np.float32)**2))

                    if energy > self.energy_threshold:
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

        if not frames or not has_spoken:
            logger.warning("No speech detected")
            return None

        # Save to temp file
        temp_file = "temp_audio.wav"
        try:
            import wave
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))
            return temp_file
        except Exception as e:
            logger.error(f"Could not save audio file: {e}")
            return None

    def transcribe(self, audio_file):
        """Transcribe audio to text"""
        if not self.whisper_model:
            return None

        start_time = time.time()

        try:
            segments, info = self.whisper_model.transcribe(
                audio_file,
                beam_size=2,
                language="en",
                vad_filter=True,
                temperature=0
            )

            text = " ".join([segment.text.strip() for segment in segments])

            duration = time.time() - start_time
            logger.info(f"⚡ Transcribed in {duration:.2f}s")

            return text.strip()

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    async def speak_async(self, text, voice="en-US-ChristopherNeural", rate="+30%"):
        """Async text-to-speech"""
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                temp_mp3 = f.name

            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(temp_mp3)

            if self.interrupt_flag.is_set():
                try:
                    os.remove(temp_mp3)
                except:
                    pass
                return

            try:
                pygame.mixer.music.load(temp_mp3)
                pygame.mixer.music.play()

                while pygame.mixer.music.get_busy():
                    if self.interrupt_flag.is_set():
                        pygame.mixer.music.stop()
                        break
                    pygame.time.wait(10)

            finally:
                pygame.mixer.music.unload()
                try:
                    os.remove(temp_mp3)
                except:
                    pass

        except Exception as e:
            logger.error(f"TTS Error: {e}")

    def speak(self, text):
        """Speak text (blocking)"""
        logger.info(f"🤖 Agent: {text}")
        self.interrupt_flag.clear()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.speak_async(text))
        finally:
            loop.close()


# ============================================================================
# VOICE AGENT - Combines AI Agent with Voice Interface
# ============================================================================

class VoiceAgent:
    """Voice-controlled AI Agent"""

    def __init__(self):
        self.voice = VoiceInterface()
        self.agent = AIAgent(
            model="llama3.2",
            enable_planning=True,
            memory_file="agent_memory.json"
        )

        logger.info("🤖 Voice Agent Ready!")

    def run(self):
        """Main loop"""
        print("\n" + "="*70)
        print("🤖 AI VOICE AGENT - Autonomous Assistant")
        print("="*70)
        print("\n💡 Instructions:")
        print("  • Press ENTER to talk")
        print("  • Ask me to perform tasks:")
        print("    - 'Search Google for Python tutorials'")
        print("    - 'Open youtube.com'")
        print("    - 'Create a file called test.txt with hello world'")
        print("    - 'List all files in current directory'")
        print("  • Type '/plan <task>' to see execution plan")
        print("  • Type '/status' to see agent status")
        print("  • Type '/reflect' for agent self-reflection")
        print("  • Type 'quit' to exit")
        print("="*70 + "\n")

        # Greeting
        self.voice.speak("Hey! I'm your AI agent. I can plan and execute complex tasks. What do you need?")

        while True:
            try:
                user_input = input("\n> ").strip()

                # Commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    self.voice.speak("Goodbye!")
                    break

                elif user_input == '/status':
                    status = self.agent.get_status()
                    print(f"\n📊 Agent Status:")
                    print(f"   Model: {status['model']}")
                    print(f"   Tools: {status['tools_available']}")
                    print(f"   Memory: {status['memory_stats']['total_count']} entries")
                    print(f"   Planning: {'Enabled' if status['planning_enabled'] else 'Disabled'}")
                    print(f"\n🔧 Available tools:")
                    for tool in status['tools']:
                        print(f"   • {tool}")

                elif user_input == '/reflect':
                    print("\n🤔 Agent reflecting...")
                    reflection = self.agent.reflect()
                    print(f"\n💭 Reflection:\n{reflection}")
                    self.voice.speak(reflection)

                elif user_input.startswith('/plan '):
                    task = user_input[6:].strip()
                    print(f"\n📋 Creating plan for: {task}")

                    # Create plan without executing
                    tools = self.agent.tool_registry.list_tools()
                    plan = self.agent.planner.create_plan(task, tools)

                    summary = self.agent.planner.get_plan_summary(plan)
                    print(f"\n{summary}")

                    self.voice.speak(f"I would execute this in {len(plan.tasks)} steps. Check the terminal for details.")

                # Voice input
                elif user_input == '':
                    audio_file = self.voice.record_audio()

                    if audio_file and os.path.exists(audio_file):
                        text = self.voice.transcribe(audio_file)

                        if text:
                            print(f"\n💬 You: {text}")

                            # Execute task with agent
                            print("\n🤖 Agent working...")
                            result = self.agent.execute_task(text, use_planning=True)

                            response = result.get('output', 'Task completed')
                            print(f"\n✓ {response}")

                            self.voice.speak(response)

                        try:
                            os.remove(audio_file)
                        except:
                            pass

                # Text input
                else:
                    print("\n🤖 Agent working...")
                    result = self.agent.execute_task(user_input, use_planning=True)

                    response = result.get('output', 'Task completed')
                    print(f"\n✓ {response}")

                    self.voice.speak(response)

            except KeyboardInterrupt:
                print("\n\n🛑 Interrupted!")
                self.voice.speak("Goodbye!")
                break

            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                print(f"❌ Error: {e}")


# ============================================================================
# MAIN
# ============================================================================

def check_dependencies():
    """Check dependencies"""
    try:
        ollama.list()
        logger.info("✓ Ollama is running")
    except Exception as e:
        logger.error(f"❌ Ollama not running: {e}")
        logger.error("Please start Ollama first: https://ollama.ai")
        return False

    return True


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("🤖 AI VOICE AGENT - Initializing...")
    print("="*70 + "\n")

    if not check_dependencies():
        sys.exit(1)

    try:
        agent = VoiceAgent()
        agent.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
