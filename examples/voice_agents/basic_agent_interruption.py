import logging
import os
import re
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    UserInputTranscribedEvent,
    AgentStateChangedEvent,
    WorkerOptions,
    cli,
    metrics,
    RoomInputOptions,
    RoomOutputOptions
)
from livekit.agents.llm import function_tool
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from basic_agent_interruption_handler import (
    InterruptionPolicy,
    InterruptionFilter,
    InterruptionDecision,
)

logger = logging.getLogger("basic-agent")
load_dotenv()

# =========================
# AGENT
# =========================
class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "Your name is Kelly. "
                "You interact with users via voice. "
                "Keep responses concise and natural. "
                "Do not use emojis, markdown, or special characters. "
                "You are friendly and curious."
            )
        )

    async def on_enter(self):
        self.session.generate_reply()

    @function_tool
    async def lookup_weather(
        self,
        context: RunContext,
        location: str,
        latitude: str,
        longitude: str,
    ):
        return "sunny with git temperature of 70 degrees."


# =========================
# PREWARM
# =========================
def prewarm(proc: JobProcess):
    # Load VAD once per worker
    proc.userdata["vad"] = silero.VAD.load()
    proc.userdata["interrupt_policy"] = InterruptionPolicy.from_env()


# =========================
# ENTRYPOINT
# =========================
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Disable framework auto-interruptions
    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-2:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",

        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],

        preemptive_generation=True,

        # CRITICAL FLAGS
        allow_interruptions=False,
        discard_audio_if_uninterruptible=False,
    )

    # =========================
    # METRICS
    # =========================
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        usage_collector.collect(ev.metrics)

    ctx.add_shutdown_callback(
        lambda: logger.info(f"Usage: {usage_collector.get_summary()}")
    )

    # =========================
    # AGENT SPEAKING STATE
    # =========================
    agent_speaking = {"value": False}

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        agent_speaking["value"] = ev.new_state == "speaking"
        logger.debug("Agent state: %s -> %s", ev.old_state, ev.new_state)

    # =========================
    # INTERRUPTION FILTER
    # =========================
    interrupt_filter = InterruptionFilter(ctx.proc.userdata["interrupt_policy"])

    # =========================
    # USER INPUT HANDLER (CORRECT EVENT)
    # =========================
    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev: UserInputTranscribedEvent):
        text = (ev.transcript or "").strip()
        confidence = getattr(ev, "confidence", None)

        if not text:
            return

        # If agent is not speaking, let framework handle normally
        if not agent_speaking["value"]:
            logger.debug("User spoke while agent silent: %r", text)
            return

        logger.info("User spoke while agent speaking: %r", text)

        decision = interrupt_filter.decide(
            text=text,
            confidence=confidence,
            agent_speaking=True,
        )

        if decision == InterruptionDecision.IGNORE:
            # Prevent fillers from becoming a user turn
            session.clear_user_turn()
            logger.info("Ignored filler/backchannel: %r", text)
            return

        if decision == InterruptionDecision.INTERRUPT:
            logger.info("Interrupting agent due to: %r", text)
            session.interrupt(force=True)
            return

    # =========================
    # START SESSION
    # =========================
    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm)
    )
