"""Nabu voice assistant main loop."""

import logging
import sys

import numpy as np

from app.logging_setup import setup_logging
from app.config import load_config
from app.audio.capture import AudioCapture
from app.audio.playback import AudioPlayback
from app.stt.whisper_stt import WhisperSTT
from app.tts.nabu_tts import NabuTTS
from app.llm.client import LLMClient, LLMError
from app.wake.detector import WakeWordDetector
from app.wake.speaker import SpeakerParser
from app.wake.conversations import ConversationManager
from app.intent.router import IntentRouter
from app.intent.handlers import (
    handle_time_date,
    handle_home_control,
    handle_web_search,
    handle_system,
    handle_memory_store,
    handle_memory_query,
    handle_volume_control,
)
from app.search.tavily import TavilyClient
from app.search.llm_search import build_nosearch_prompt
from app.llm.sentence_splitter import split_sentences
from app.memory.sqlite_store import MemorySQLite
from app.memory.vector_store import MemoryVectorStore
from app.memory.extractor import MemoryExtractor
from app.homeassistant.client import HAClient
from app.homeassistant.shortcuts import ShortcutHandler

setup_logging()
log = logging.getLogger("nabu")


def generate_beep(frequency=800, duration=0.15, sample_rate=16000):
    """Generate a short sine wave beep as float32 array."""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    fade_samples = int(0.01 * sample_rate)
    envelope = np.ones_like(t)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    return 0.5 * np.sin(2 * np.pi * frequency * t) * envelope


def _listen_short(capture, stt, sample_rate, wake_vad):
    """Record a short utterance and transcribe. Returns transcript or None."""
    audio = capture.record_utterance(
        silence_duration=wake_vad["silence_duration"],
        max_duration=wake_vad["max_duration"],
    )
    if len(audio) / sample_rate < 0.3:
        return None
    transcript, _ = stt.transcribe(audio)
    return transcript


def _speak(tts, playback, text):
    """Synthesize and play text (streaming when HAL is available)."""
    log.info("Speaking: %s", text)
    tts.speak(text, playback)


def main():
    config = load_config()
    llm_cfg = config["llm"]
    wake_cfg = config["wake"]
    wake_vad = wake_cfg["vad"]
    ack_cfg = wake_cfg["acknowledgment"]
    sample_rate = config["audio"]["sample_rate"]
    data_cfg = config.get("data", {})

    log.info("Loading STT model...")
    stt = WhisperSTT(config)
    log.info("Loading TTS engine...")
    tts = NabuTTS(config)
    capture = AudioCapture(config)
    playback = AudioPlayback(config)
    llm = LLMClient(config)
    tavily = TavilyClient(config)

    memory_cfg = config.get("memory", {})
    if memory_cfg.get("enabled", False):
        mem_db = MemorySQLite(memory_cfg.get("db_path", "data/memory/nabu_memory.db"))
        mem_vec = MemoryVectorStore(
            db_path=memory_cfg.get("vector_path", "data/memory/lancedb"),
            model_name=memory_cfg.get("embedding_model"),
        )
        extractor = MemoryExtractor(mem_db, mem_vec)
    else:
        extractor = None

    ha_client = HAClient(config)
    shortcut_handler = ShortcutHandler(config, ha_client)

    detector = WakeWordDetector(config)
    speaker_parser = SpeakerParser(config)
    router = IntentRouter(config)

    conv_dir = data_cfg.get("conversations_dir", "data/conversations")
    conv_mgr = ConversationManager(
        data_dir=conv_dir,
        max_context_tokens=llm_cfg.get("max_context_tokens", 28000),
    )

    base_prompt = llm_cfg["system_prompt"]

    beep = generate_beep(ack_cfg["beep_frequency"], ack_cfg["beep_duration"], sample_rate)
    ready_beep = generate_beep(600, 0.1, sample_rate)

    if llm.health_check_sync():
        log.info("LLM server reachable: %s", llm_cfg["model"])
    else:
        log.warning("LLM server not reachable at %s", llm_cfg["url"])

    if tavily.enabled:
        log.info("Web search enabled (Tavily)")

    if extractor and extractor.enabled:
        log.info("Memory system enabled (SQLite + LanceDB)")

    if ha_client.enabled:
        log.info("Home Assistant connected (%s)", config.get("homeassistant", {}).get("url"))

    log.info("Ready. Listening for 'OK, Nabu!'. Press Ctrl+C to exit.\n")
    _speak(tts, playback, "Nabu ist bereit.")

    try:
        while True:
            # Listen for wake word
            audio = capture.record_utterance(
                silence_duration=wake_vad["silence_duration"],
                max_duration=wake_vad["max_duration"],
            )
            duration = len(audio) / sample_rate
            rms = float(np.sqrt(np.mean(audio[:min(len(audio), 4800)] ** 2))) if len(audio) > 0 else 0
            log.debug("Audio captured: %.2fs, %d samples, RMS=%.4f", duration, len(audio), rms)
            if duration < 0.3:
                log.debug("Too short (%.2fs < 0.3s), skipping", duration)
                continue

            transcript, _ = stt.transcribe(audio)
            log.debug("Wake check: '%s' (%.2fs audio)", transcript, duration)

            if not detector.check(transcript):
                log.debug("No wake word in: '%s'", transcript)
                continue

            shortcut_result = shortcut_handler.check(transcript)
            if shortcut_result:
                log.info("Shortcut executed from wake transcript")
                _speak(tts, playback, shortcut_result["response"])
                continue

            user_id = speaker_parser.parse(transcript)

            if user_id is None:
                _speak(tts, playback, f"Wer spricht? {speaker_parser.speaker_names_list()}?")

                id_text = _listen_short(capture, stt, sample_rate, wake_vad)
                if id_text is None:
                    log.debug("No response, returning to wake word listening")
                    continue

                log.debug("Speaker ID response: '%s'", id_text)
                user_id = speaker_parser.parse(id_text)

                if user_id is None:
                    log.info("Could not identify speaker from: '%s'", id_text)
                    _speak(tts, playback, "Ich konnte dich leider nicht erkennen. Versuche es nochmal mit 'OK, Nabu'.")
                    continue

            display_name = speaker_parser.display_name(user_id)
            log.info("Wake word detected! Speaker: %s", display_name)

            if conv_mgr.has_conversations(user_id):
                _speak(tts, playback,
                       f"Hallo {display_name}! Möchtest du eine Diskussion fortsetzen oder geht es um ein neues Thema?")

                choice_text = _listen_short(capture, stt, sample_rate, wake_vad)
                if choice_text is not None:
                    log.debug("Topic choice: '%s'", choice_text)
                    choice_lower = choice_text.lower()

                    if any(w in choice_lower for w in ["fortsetz", "weiter", "alte", "letzte"]):
                        topic_list = conv_mgr.format_topic_list(user_id)
                        topics = conv_mgr.list_topics(user_id)

                        if len(topics) == 1:
                            _speak(tts, playback, f"Alles klar, wir machen weiter mit: {topics[0]}")
                        else:
                            _speak(tts, playback,
                                   f"Hier sind deine bisherigen Gespräche: {topic_list}. Welche Nummer?")

                            num_text = _listen_short(capture, stt, sample_rate, wake_vad)
                            if num_text is not None:
                                log.debug("Selection: '%s'", num_text)
                                if not conv_mgr.select_conversation(user_id, num_text):
                                    _speak(tts, playback, "Das habe ich nicht verstanden. Wir starten ein neues Thema.")
                                    conv_mgr.start_new(user_id)
                            else:
                                conv_mgr.start_new(user_id)
                    else:
                        conv_mgr.start_new(user_id)
                        log.info("New conversation started for %s", display_name)
                else:
                    conv_mgr.start_new(user_id)

            playback.play_array(beep, sample_rate)

            conv_cfg = config.get("conversation", {})
            idle_timeout = conv_cfg.get("idle_timeout", 30.0)
            max_conv_duration = conv_cfg.get("max_duration", 300.0)

            import time as _time
            conv_start = _time.monotonic()
            silent_rounds = 0
            max_silent_rounds = int(idle_timeout / (config["audio"]["vad"]["silence_duration"]))

            log.info("Conversation started with %s (timeout=%ds)", display_name, int(idle_timeout))

            while True:
                elapsed = _time.monotonic() - conv_start
                if elapsed > max_conv_duration:
                    log.info("Conversation max duration reached (%.0fs)", elapsed)
                    _speak(tts, playback, "Wir sprechen schon eine Weile. Ich gehe wieder in den Ruhemodus.")
                    break

                log.debug("Waiting for next command from %s (%.0fs into conversation)...", display_name, elapsed)
                command_audio = capture.record_utterance()
                command_duration = len(command_audio) / sample_rate

                if command_duration < 0.3:
                    silent_rounds += 1
                    log.debug("No speech detected (%d/%d silent rounds)", silent_rounds, max_silent_rounds)
                    if silent_rounds >= max_silent_rounds:
                        log.info("Idle timeout reached (%ds silence), asking user...", int(idle_timeout))
                        _speak(tts, playback, "Ich habe nichts mehr gehört. Möchtest du das Gespräch beenden?")

                        followup = _listen_short(capture, stt, sample_rate, wake_vad)
                        if followup:
                            followup_lower = followup.lower()
                            log.debug("Timeout response: '%s'", followup)
                            if any(w in followup_lower for w in ["nein", "warte", "noch nicht", "weiter", "bleib"]):
                                log.info("User wants to continue conversation")
                                _speak(tts, playback, "Alles klar, ich höre weiter zu.")
                                silent_rounds = 0
                                continue

                        log.info("Conversation ending after timeout")
                        _speak(tts, playback, f"Vielen Dank für das Gespräch, {display_name}!")
                        break
                    continue

                silent_rounds = 0  # reset on any speech

                command_text, language = stt.transcribe(command_audio)
                log.debug("Command STT: [%s] '%s' (%.2fs audio)", language, command_text, command_duration)
                log.info("[%s] %s", language, command_text)

                if not command_text.strip():
                    log.debug("Empty transcript, skipping")
                    continue

                intent = router.classify(command_text)
                log.info("Intent: %s → '%s'", intent, command_text)

                if intent == "end_conversation":
                    log.info("User ended conversation: '%s'", command_text)
                    _speak(tts, playback, f"Alles klar, bis später {display_name}!")
                    break

                if intent == "delete_conversations":
                    _speak(tts, playback, "Bist du sicher? Alle deine Konversationen und Erinnerungen werden gelöscht. Ja oder nein?")
                    confirm = _listen_short(capture, stt, sample_rate, wake_vad)
                    if confirm and any(w in confirm.lower() for w in ["ja", "yes", "sicher", "mach"]):
                        conv_mgr.delete_all(user_id)
                        if extractor:
                            extractor.delete_all_for_user(user_id)
                        log.info("All data deleted for %s (conversations + memories)", display_name)
                        _speak(tts, playback, "Alles klar, alle deine Konversationen und Erinnerungen wurden gelöscht.")
                    else:
                        _speak(tts, playback, "Abgebrochen. Deine Konversationen bleiben erhalten.")
                    playback.play_array(ready_beep, sample_rate)
                    continue

                if intent == "volume_control":
                    response_text = handle_volume_control(command_text, playback=playback)
                    _speak(tts, playback, response_text)
                    playback.play_array(ready_beep, sample_rate)
                    continue

                history = conv_mgr.get_active_history(user_id)

                if not history:
                    conv_mgr.set_topic(user_id, command_text)

                already_spoken = False

                if intent == "time_date":
                    response_text = handle_time_date(command_text)
                    history.append({"role": "user", "content": command_text})
                    history.append({"role": "assistant", "content": response_text})
                    conv_mgr.save(user_id)

                elif intent == "home_control":
                    response_text = handle_home_control(command_text)
                    history.append({"role": "user", "content": command_text})
                    history.append({"role": "assistant", "content": response_text})
                    conv_mgr.save(user_id)

                elif intent == "web_search":
                    _speak(tts, playback, "Moment, ich suche...")
                    response_text = handle_web_search(
                        command_text, tavily=tavily, llm=llm,
                        base_system_prompt=base_prompt,
                    )
                    history.append({"role": "user", "content": command_text})
                    history.append({"role": "assistant", "content": response_text})
                    conv_mgr.save(user_id)

                elif intent == "system":
                    response_text = handle_system(
                        command_text, display_name=display_name, llm=llm,
                    )
                    history.append({"role": "user", "content": command_text})
                    history.append({"role": "assistant", "content": response_text})
                    conv_mgr.save(user_id)

                elif intent == "memory_store":
                    response_text = handle_memory_store(
                        command_text, extractor=extractor, llm=llm,
                        user_id=user_id, display_name=display_name,
                    )
                    history.append({"role": "user", "content": command_text})
                    history.append({"role": "assistant", "content": response_text})
                    conv_mgr.save(user_id)

                elif intent == "memory_query":
                    response_text = handle_memory_query(
                        command_text, extractor=extractor, user_id=user_id,
                    )
                    history.append({"role": "user", "content": command_text})
                    history.append({"role": "assistant", "content": response_text})
                    conv_mgr.save(user_id)

                else:
                    history.append({"role": "user", "content": command_text})
                    llm_messages = conv_mgr.get_history_for_llm(user_id)

                    memory_context = ""
                    if extractor:
                        try:
                            memory_context = extractor.retrieve_relevant(command_text, user_id)
                        except Exception as e:
                            log.warning("Memory retrieval failed: %s", e)

                    user_prompt = build_nosearch_prompt(
                        display_name, memory_context,
                        base_system_prompt=base_prompt,
                    )

                    try:
                        tokens = llm.stream_tokens_sync(
                            system_prompt=user_prompt,
                            messages=llm_messages,
                        )
                        sentences = split_sentences(tokens)
                        response_text = tts.speak_streamed(sentences, playback)
                        already_spoken = True

                        log.info("Speaking: %s", response_text)
                        history.append({"role": "assistant", "content": response_text})
                        conv_mgr.save(user_id)

                        if extractor and memory_cfg.get("extract_after_turns", True):
                            try:
                                extracted = extractor.extract_and_store(
                                    user_id, command_text, response_text, llm,
                                    display_name=display_name,
                                )
                                if extracted:
                                    log.info("Extracted %d memories", len(extracted))
                            except Exception as e:
                                log.warning("Memory extraction failed: %s", e)

                        if conv_mgr.needs_summary(user_id):
                            try:
                                summary = llm.complete_sync(
                                    system_prompt="Fasse dieses Gespräch in 8 bis 10 Wörtern zusammen, nur das Thema:",
                                    messages=history[-2:],
                                    max_tokens=50,
                                    temperature=0.3,
                                )
                                conv_mgr.update_topic(user_id, summary.strip().rstrip('.'))
                                log.info("Topic summary: %s", summary.strip())
                            except LLMError:
                                pass

                    except LLMError as e:
                        log.error("LLM error: %s", e)
                        response_text = "Der Sprachserver ist gerade nicht erreichbar. Bitte versuche es gleich nochmal."
                        history.pop()

                if not already_spoken:
                    _speak(tts, playback, response_text)
                playback.play_array(ready_beep, sample_rate)

            log.info("Conversation with %s ended. Back to wake word listening.\n", display_name)
            playback.play_array(ready_beep, sample_rate)

    except KeyboardInterrupt:
        log.info("Shutting down")
        llm.close_sync()
        tts.close()
        if memory_cfg.get("enabled", False):
            mem_db.close()
        sys.exit(0)


if __name__ == "__main__":
    main()
