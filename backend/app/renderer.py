import os
import logging
import subprocess
import tempfile
import glob
import shutil
from .config import settings
from .generator import generate_manim_code
from .supabase_client import update_job_data, upload_to_supabase
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from langchain_cohere import ChatCohere
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from io import BytesIO
import wave
import math
from pydub import AudioSegment
import re
import json
import math
from cartesia import Cartesia,AsyncCartesia
import asyncio
from pydantic import BaseModel, Field
from typing import List,Optional
import ffmpeg

from strands import Agent,tool
from strands.multiagent import Swarm
from strands.models.litellm import LiteLLMModel
from strands_tools import calculator # Import the calculator tool
from typing import Any, Dict

model = LiteLLMModel(
    client_args={
        "api_key":settings.OPENROUTER_API_KEY,
    },
    model_id="openrouter/openai/gpt-4o-mini",
    # model_id="openrouter/google/gemini-2.0-flash-lite-001",
    # model_id="openrouter/google/gemini-2.0-flash-exp:free",
    # model_id="openrouter/google/gemini-2.0-flash-001",
    # model_id="openrouter/google/gemini-2.5-pro",
    params={
        'temperature':0.5,
        "max_tokens":1000
    },
)

# agent = Agent(
#     model=model,
#     tools=[calculator,smart_background_agents],
#     system_prompt="You are a voice ai agent in realtime meeting with user,use tool 'smart_backgroud_agents' who will help you in the backgroud with info and task you delegate to them, DO NOT tell user about this and meantime try to answer by yourself you will know when the answer comes until them you have to handle user without that info and without letting him know that you dont have the information yet, and also in your resposnes to user add fillers like um hm or any more where appropriate to make it sound more humanistic"
# )

class TranscriptPhase(BaseModel):
    phase_id: int = Field(
        description="Sequential phase number starting from 1"
    )

    voiceover_text: str = Field(
        description="Exact narration text to be spoken by the voiceover"
    )

    visual_instruction: str = Field(
        description=(
            "Detailed Manim visual instructions describing objects, "
            "positions, labels, colors, and spatial relationships"
        )
    )

    animation_type: Optional[str] = Field(
        default=None,
        description="Manim animation type used to introduce or modify visuals, keep it '' if no animation_type given"
    )

    duration_seconds: Optional[float] = None




class ManimSynchronizedTranscript(BaseModel):
    """
    The complete plan for generating a synchronized Manim + voiceover video.
    """

    phases: List[TranscriptPhase] = Field(
        description="Ordered list of synchronized narration and visual phases"
    )


class TTSFinalTranscript(BaseModel):
    """
    Phase-by-phase TTS-ready transcripts with embedded SSML timing.
    Each phase corresponds to one visual scene in the Manim animation.
    """
    phasewise_transcripts: List[str] = Field(
        description=(
            "List of TTS-ready strings, one per phase. Each string contains "
            "the voiceover text with SSML <break> tags for natural pacing. "
            "Order matches the phase_id sequence from ManimSynchronizedTranscript."
        )
    )


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RENDER_DIR = settings.RENDER_DIR
MAX_ITERATIONS = 5

chat = ChatCohere(
    model="command-r-plus-08-2024", 
    verbose=True,
    cohere_api_key=settings.COHERE_API_KEY,
    temperature=0.3 
)




llm_with_structured_output = chat.with_structured_output(ManimSynchronizedTranscript)

def run_manim(code: str, temp_dir: str, quality: str = "m") -> tuple[bool, str]:
    """
    Run Manim code and render the animation in a temporary directory.

    Args:
        code: The Python code to render
        temp_dir: Directory to save the rendered video temporarily
        quality: Render quality (l=low, m=medium, h=high)

    Returns:
        Tuple of (success, error_message or output_file)
    """

    temp_path = None
    try:
        os.makedirs(temp_dir, exist_ok=True)
        logger.info(f"Using temporary directory: {temp_dir}")

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".py", mode="w", delete=False
            ) as temp_file:
                temp_file.write(code)
                temp_path = temp_file.name
                logger.info(f"Created temporary file: {temp_path}")
        except Exception as e:
            logger.error(f"Failed to create temporary file: {str(e)}")
            return False, f"Failed to create temporary file: {str(e)}"

        cmd = [
            "python",
            "-m",
            "manim",
            temp_path,
            "Scene",
            f"-q{quality}",
            "--format=mp4",
            f"--media_dir={temp_dir}",
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        try:
            process = subprocess.run(
                cmd, capture_output=True, text=True, check=False, timeout=settings.MANIM_TIMEOUT
            )

        except subprocess.TimeoutExpired:
            logger.error(f"Manim rendering timed out after {settings.MANIM_TIMEOUT} seconds")
            return False, f"Rendering timed out after {settings.MANIM_TIMEOUT} seconds"

        logger.info(f"Manim stdout: {process.stdout}")
        logger.error(f"Manim stderr: {process.stderr}")

        if process.returncode != 0:
            logger.error(f"Manim rendering failed with exit code {process.returncode}")
            return False, process.stderr

        mp4_files = [
            f
            for f in glob.glob(os.path.join(temp_dir, "**", "*.mp4"), recursive=True)
            if "partial_movie_files" not in f
        ]
        logger.info(f"Found MP4 files: {mp4_files}")

        if not mp4_files:
            logger.error("No MP4 file found after rendering")
            return False, "No MP4 file found after rendering"

        output_file = mp4_files[0]

        logger.info(f"Manim rendering completed successfully: {output_file}")
        return True, output_file

    except Exception as e:
        error_msg = f"Error running Manim: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.info(f"Removed temporary file: {temp_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to clean up temporary file {temp_path}: {str(e)}"
                )

def normalize_word(text):
    """
    Removes punctuation and converts to lowercase for easy matching.
    Example: "Let's!" -> "lets"
    """
    return re.sub(r'[^\w]', '', text).lower()

def map_timestamps_to_phases(phases_json, cartesia_timestamps):
    """
    Inputs:
        phases_json: List of dicts (from your LLM Step 2)
        cartesia_timestamps: List of dicts [{'word': 'Hello', 'start': 0.1, 'end': 0.3}, ...]
    
    Returns:
        The phases_json with a new 'audio_duration' key in each object.
    """
    
    current_ts_index = 0
    total_timestamps = len(cartesia_timestamps)

    for phase in phases_json:
        text = phase.get('voiceover_text', "")
        
        # 1. Clean the text into a list of checkable words
        # "Let's start." -> ["lets", "start"]
        target_words = [normalize_word(w) for w in text.split()]
        
        # Filter out empty strings
        target_words = [w for w in target_words if w]

        if not target_words:
            # If phase has no text (just silence), default to 2s or 0s
            phase['audio_duration'] = 2.0
            continue

        # 2. Capture Start Time
        # The start of this phase is the start time of the next available word in the stream
        if current_ts_index < total_timestamps:
            start_time = cartesia_timestamps[current_ts_index]['start']
        else:
            phase['audio_duration'] = 2.0
            continue

        # 3. Advance the cursor through the timestamp list
        # We look for the words in this phase to "consume" them from the master list
        matches_found = 0
        
        for target_word in target_words:
            # Search forward in the timestamp list until we find a match
            # This handles cases where Cartesia might split words differently
            while current_ts_index < total_timestamps:
                ts_word = normalize_word(cartesia_timestamps[current_ts_index]['word'])
                current_ts_index += 1
                
                if ts_word == target_word: # Found a match!
                    matches_found += 1
                    break # Move to next target word
                
                # If words don't match, we skip the timestamp word (it might be a filler or noise)
        
        # 4. Capture End Time
        # The end of this phase is the 'end' time of the LAST word we consumed
        # We use (current_ts_index - 1) because the loop incremented it one extra time
        if current_ts_index > 0:
            end_time = cartesia_timestamps[current_ts_index - 1]['end']
        else:
            end_time = start_time + 2.0

        # 5. Calculate Duration & Add Buffer
        duration = end_time - start_time
        
        # CRITICAL: Add 0.5s buffer so the animation doesn't snap instantly to the next one
        phase['audio_duration'] = round(duration + 0.5, 2)

    return phases_json


async def process_rendering_job(job_id: str, prompt: str, quality: str):
    """
    Process a rendering job from start to finish with iterative error correction:
    1. Generate Manim code
    2. Run Manim to create animation in a temp directory
    3. If error occurs, try to fix up to MAX_ITERATIONS times
    4. Upload successful result to Supabase
    5. Clean up temp files
    """
    job_dir = os.path.join(RENDER_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    conversation_history = []
    conversation_history.append(HumanMessage(content=prompt))

    #prompt = user_query
    #1.create one transcript for tutoring
    #2.manim_scynchronized_transcript(generates transcript as well as mentions what the manim will be showing in the video)
    #3.TTS takes the manim_sychcronized_transcript and create one FINAL transcript with proper pauses and all and generates on entire audio
    #4.Give the manim_synchronized_transcript as well as TTS FINAL Transcript to manim code generator so it can adopt to it
    
    @tool(context=True)
    def set_tutor_transcript(new_tutor_transcript:str,tool_context: Dict[str, Any]):
        """
        Tool used to set the tutor transcript on which
    manim_synschornized_transcript and final_tts_transcript_phasewise
    will be created.

        Args:
            new_tutor_transcript: The updated tutor transcript text.
        """

        state = tool_context.get("invocation_state", {})
    
        # Initialize state if missing
        state.setdefault("tutor_transcript", "")
        
        state["tutor_transcript"]=new_tutor_transcript
        
        # Write back to invocation_state so the mutation is preserved
        tool_context[""] = state

        return "tutor_transcript updated successfully"
    

    @tool(context=True)
    def set_manim_synchronized_transcript(new_manim_synchronized_transcript:List[Any],tool_context: Dict[str, Any]):
        """Tool used to set the manim_synchronized_transcripts , all at once , previous once are overwritten 

        Args:
        new_manim_synchronized_transcript :List[Any] : ex [{
            "phase_id": 1,
            "voiceover_text": "Let's start with a right-angled triangle.",
            "visual_instruction": "Create a Right Triangle in the center. Labels: 'a' (bottom), 'b' (left), 'c' (hypotenuse). Style: White lines.",
            "animation_type": "Create/Write"
        },
        {
            "phase_id": 2,
            "voiceover_text": "Now, we attach a square to side 'a'.",
            "visual_instruction": "Create a Square. Position: Attached to the BOTTOM edge of the triangle. Color: Green with Black Fill. Label: 'a²' inside the square.",
            "animation_type": "GrowFromEdge"
        }
        ]

        """

        state = tool_context.get("invocation_state", {})
    
        # Initialize state if missing
        state.setdefault("manim_synchronized_transcript", [])
        
        state["manim_synchronized_transcript"]=new_manim_synchronized_transcript
        
        tool_context[""] = state

        return "manim_synchronized_transcript updated successfully"

    @tool(context=True)
    def set_tts_phasewise_transcript(new_tts_phasewise_transcript:list[str],tool_context: Dict[str, Any]):
        """ Tool used to set tts_phasewise_transcript to the state
            previous state will be overwritten so set all at once
        Args:
            new_tts_phasewise:list[str] : ex = [
                "Let's start with a right-angled triangle. <break time='1.0s'/>",
                "Now we attach a square to each side <break time='0.4s'/> and see something amazing. <break time='1.0s'/>"
            ]
        """

        state = tool_context.get("invocation_state", {})
    
        # Initialize state if missing
        state.setdefault("tts_phasewise_transcript", [])
        
        state["tts_phasewise_transcript"]=new_tts_phasewise_transcript
        
        tool_context[""] = state

        return "tts_phasewise_transcript updated successfully"


    # Step 2: Create the swarm (agents coordinate autonomously)
  

    tutor_transcript_generator_system_prompt = """
    You are an expert Educational Scriptwriter for short, animated explainer videos. 
    Your task is to take a simple user topic (e.g., "Area of a Triangle") and generate a high-quality, engaging voiceover transcript.

    ### YOUR GOAL
    Write a clear, concise, and conversational script that a Text-to-Speech (TTS) engine will read. The script must explain the concept step-by-step.

    ### CRITICAL WRITING RULES:
    1.  **Pure Spoken Audio Only:** Do NOT include scene descriptions, camera directions, or visual cues like [Draw Triangle] or (Pause). Write ONLY what the voice should say.
    2.  **"Visual-Ready" Language:** Write as if the viewer is looking at the screen. Use pointing language:
        * *Good:* "Look at this shape here." / "Notice how the height connects to the base."
        * *Bad:* "Imagine a triangle." (No, we are showing it).
        * *Bad:* "I am now drawing a red line." (Don't describe the action, explain the concept).
    3.  **Pacing:** Break the text into short, logical paragraphs. Each paragraph will eventually become a distinct animation phase.
    4.  **Tone:** Enthusiastic, clear, and beginner-friendly. Avoid overly complex jargon unless you explain it.
    5.  **Length:** Keep it focused. Target a duration of 30-60 seconds (approx. 75-150 words).

    ### OUTPUT FORMAT
    Return the transcript as plain text, separated by double newlines for logical pauses.

    ### EXAMPLE INPUT:
    "Explain the Pythagorean Theorem"

    ### EXAMPLE OUTPUT:
    "Let's look at a right-angled triangle. This creates a unique relationship between its three sides.

    We call the two shorter sides 'a' and 'b', and the longest side, opposite the right angle, is the hypotenuse, 'c'.

    Now, imagine we build a square on each of these sides. The theorem tells us something fascinating about their areas.

    The area of square 'a' plus the area of square 'b' is exactly equal to the area of square 'c'. This is why we say a-squared plus b-squared equals c-squared."
    """


    manim_synchronized_transcript_system_prompt = """
        You are the **Visual Director** for an automated video generation pipeline. 
        Your input is an educational voiceover transcript.
        Your output is a structured **JSON** directive that maps every chunk of audio to a specific, concrete visual instruction for a Manim animator.

        ### YOUR CORE RESPONSIBILITY
        The Manim Animator (the next agent) is a blind coder. It does not understand "show the concept." 
        You must tell it EXACTLY:
        1. **WHAT** to draw (Shape, Color, Label).
        2. **WHERE** to place it (Coordinates, Relative Position).
        3. **HOW** to move it (Animation type).

        ### CRITICAL RULES FOR VISUAL INSTRUCTIONS

        1.  **Layout & Flow (The "Anti-Overlap" Rule):**
            * Establish a visual flow (usually Left-to-Right).
            * **Explicit Positioning:** Never say "place it next to it." Say "Position this group to the RIGHT of the [Previous Object] with a buffer of 2.0 units."
            * **Memory:** Keep track of what is on screen. Do not ask to create an object that already exists. Refer to existing objects by name.

        2.  **Container Style (The "Transparency" Fix):**
            * If the visual involves a Box, Circle, or Container, explicitly instruct: "Style: Fill Color BLACK, Opacity 1.0, colored stroke."
            * This prevents lines from showing through objects.

        3.  **Labeling Strategy:**
            * Instruct the animator to place labels **OUTSIDE** objects (Above/Below), never inside, to leave room for animations.
            * If multiple labels are needed, instruct them to **STACK** (e.g., "Place label B above Label A").

        4.  **Geometry & Attachment:**
            * If attaching shapes (e.g., squares on a triangle), specify the **Exact Edge** (e.g., "Attach to the Hypotenuse/Slanted Edge", "Attach to the Bottom Edge").

        ### JSON OUTPUT FORMAT
        You must return a raw JSON list of objects. Each object represents one "Scene Phase".

        ```json
        [
        {
            "phase_id": 1,
            "voiceover_text": "Let's start with a right-angled triangle.",
            "visual_instruction": "Create a Right Triangle in the center. Labels: 'a' (bottom), 'b' (left), 'c' (hypotenuse). Style: White lines.",
            "animation_type": "Create/Write"
        },
        {
            "phase_id": 2,
            "voiceover_text": "Now, we attach a square to side 'a'.",
            "visual_instruction": "Create a Square. Position: Attached to the BOTTOM edge of the triangle. Color: Green with Black Fill. Label: 'a²' inside the square.",
            "animation_type": "GrowFromEdge"
        }
        ]```

        INPUT TRANSCRIPT:
        {transcript}

        OUTPUT:
        Generate ONLY the valid JSON list. 

    """

    tts_final_transcript_generator_system_prompt = """
       You are the **TTS Transcript Optimizer** for an educational video pipeline.

        ### YOUR ROLE
        Transform phase-by-phase educational narration into TTS-ready transcripts with precise timing controls.

        ### INPUT
        You receive a `ManimSynchronizedTranscript` object containing:
        - `phase_id`: Sequential phase number
        - `voiceover_text`: The raw narration for this phase
        - `visual_instruction`: Description of what's being animated (used to determine pause duration)
        - `animation_type`: Type of Manim animation (e.g., Create, Write, Transform)

        ### YOUR TASK
        Generate a list of TTS-ready strings, ONE STRING PER PHASE, with SSML break tags inserted for natural pacing.

        ### CRITICAL RULES

        1. **One-to-One Mapping**: Output exactly ONE transcript string for each input phase, in the same order.

        2. **SSML Break Placement**: Insert `<break>` tags WITHIN each phase based on:
        - **Sentence boundaries**: Add `<break time="0.4s"/>` after sentences
        - **Clause breaks**: Add `<break time="0.3s"/>` after commas or logical pauses
        - **Complex visuals**: If `visual_instruction` mentions multiple objects or transformations, add `<break time="0.6s"/>` after key statements
        - **End-of-phase**: Add `<break time="1.0s"/>` at the END of each phase string to allow the animation to complete

        3. **Timing Guidelines**:
        - Simple animations (Create, Write): 0.3-0.5s pauses
        - Complex animations (Transform, Multiple objects): 0.6-1.0s pauses
        - End of phase: Always 1.0s minimum

        4. **Natural Speech Flow**: 
        - Break long sentences into digestible chunks with micro-pauses
        - Don't over-pause—maintain conversational rhythm
        - Add emphasis pauses before key concepts

        5. **Output Format**: 
        - Return ONLY the structured data (will be handled by `with_structured_output`)
        - Each string is pure TTS content—no phase numbers, no metadata
        - Do NOT wrap in `<speak>` tags (TTS engine handles that)

        ### EXAMPLE

        **Input Phases:**
        ```
        Phase 1:
        voiceover_text: "Let's start with a right-angled triangle."
        visual_instruction: "Create a Right Triangle in center with labels a, b, c"
        animation_type: "Create"

        Phase 2:
        voiceover_text: "Now we attach a square to each side and see something amazing."
        visual_instruction: "Create 3 Squares attached to each edge, colored differently"
        animation_type: "GrowFromEdge"
        ```

        **Correct Output:**
        ```json
        {
        "phasewise_transcripts": [
            "Let's start with a right-angled triangle. <break time='1.0s'/>",
            "Now we attach a square to each side <break time='0.4s'/> and see something amazing. <break time='1.0s'/>"
        ]
        }
        ```

        ### WHAT TO AVOID
        - ❌ Combining multiple phases into one string
        - ❌ Adding conversational filler ("Here's the transcript...")
        - ❌ Including phase numbers in the output
        - ❌ Forgetting the end-of-phase pause
        - ❌ Using markdown code blocks

        ### REMEMBER
        Your output will be directly fed to a TTS engine. Every word you include will be spoken. Every pause you add will be heard.
        """
    

    #new system prompts

    tutor_transcript_generator_system_prompt = """
        You are an **Expert Educational Scriptwriter** for short, animated explainer videos.

        ## 🎯 YOUR MISSION

        Generate a voiceover transcript and **call the `set_tutor_transcript` tool** to save it to the shared state.

        ---

        ## ✍️ TRANSCRIPT WRITING RULES

        ### 1. Pure Spoken Content Only
        - Write ONLY what the narrator says
        - NO stage directions: ❌ "[Draw Triangle]", ❌ "(Pause here)"
        - NO scene descriptions: ❌ "Now I'm going to draw..."

        ### 2. Visual-Ready Language (Assume viewer is watching)
        ✅ **Good:**
        - "Look at this shape here."
        - "Notice how the height connects to the base."
        - "See how these sides are equal."

        ❌ **Bad:**
        - "Imagine a triangle." (Don't imagine, it's on screen!)
        - "Let me draw a line." (Don't narrate actions)

        ### 3. Structure for Phases
        - Break into **short paragraphs** (2-4 sentences each)
        - Each paragraph = one future animation phase
        - Use double newlines between paragraphs

        ### 4. Tone
        - Enthusiastic but clear
        - Beginner-friendly
        - Conversational

        ### 5. Length
        - **30-60 seconds** of speech
        - **75-150 words** approximately

        ---

        ## 📤 WHAT YOU MUST DO

        1. **Generate** the educational voiceover transcript
        2. **Call `set_tutor_transcript(new_tutor_transcript="your transcript here")`**
        3. **Done** - the tool saves it to state, next agent will use it

        ---

        ## 📋 EXAMPLE

        **User asks:** "Explain the Pythagorean Theorem"

        **You generate this transcript and call the tool:**
        ```
        Let's look at a right-angled triangle. This creates a unique relationship between its three sides.

        We call the two shorter sides 'a' and 'b', and the longest side, opposite the right angle, is the hypotenuse, 'c'.

        Now, imagine we build a square on each of these sides. The theorem tells us something fascinating about their areas.

        The area of square 'a' plus the area of square 'b' is exactly equal to the area of square 'c'. This is why we say a-squared plus b-squared equals c-squared.
        ```

        **Then immediately call:**
        `set_tutor_transcript(new_tutor_transcript="Let's look at a right-angled triangle...")`

        ---

        **Now generate the transcript and call the tool.**
        """

    manim_synchronized_transcript_system_prompt = """
        You are the **Visual Director** for automated video generation.

        ## 🎯 YOUR MISSION

        Read the `tutor_transcript` from state, break it into visual phases, and **call the `set_manim_synchronized_transcript` tool** with a list of phase objects.

        ---

        ## 📥 INPUT (from state)

        The previous agent saved `tutor_transcript` - read it from the conversation context or state.

        ---

        ## 🎨 PHASE STRUCTURE

        Each phase is a dictionary with:

        ```python
        {
            "phase_id": 1,  # Sequential number
            "voiceover_text": "exact text from tutor_transcript for this phase",
            "visual_instruction": "EXPLICIT instructions for Manim code generator",
            "animation_type": "Create"  # Manim animation type
        }
        ```

        ---

        ## 🔧 VISUAL INSTRUCTION RULES

        ### 1. Be EXPLICIT (the code generator is blind)
        ✅ **Good:**
        - "Create a Rectangle at ORIGIN. Width: 4, Height: 2. Style: BLUE stroke width 3, BLACK fill opacity 1.0. Label 'Rectangle' placed ABOVE with 0.5 buffer."

        ❌ **Vague:**
        - "Draw a rectangle and label it"

        ### 2. Container Styling (ALWAYS include)
        For Rectangle, Circle, Square:
        ```
        "Style: Fill Color BLACK, Fill Opacity 1.0, Stroke Color [COLOR], Stroke Width 3"
        ```

        ### 3. Label Placement
        - Labels go **OUTSIDE** objects
        - "Label 'A' placed ABOVE the box with 0.3 buffer"
        - NOT "Label 'A' inside the box"

        ### 4. Positioning
        - Use specific directions: "to the RIGHT of", "BELOW", "at coordinates (2, 0, 0)"
        - NOT "next to it" or "nearby"

        ### 5. Edge Specification
        When attaching shapes:
        - "Attach to the BOTTOM edge of triangle"
        - "Align top of square with bottom of triangle"

        ---

        ## 🎬 ANIMATION TYPES

        | Use Case | Animation Type |
        |----------|---------------|
        | Draw shapes | `Create` |
        | Write text | `Write` |
        | Morph objects | `Transform` |
        | Grow from edge | `GrowFromEdge` |
        | Emphasize | `Indicate` |
        | No visual change | `NoChange` |

        ---

        ## 📤 WHAT YOU MUST DO

        1. **Read** the tutor_transcript (from state or context)
        2. **Break it** into logical phases
        3. **Create** a list of phase dictionaries (as shown above)
        4. **Call `set_manim_synchronized_transcript(new_manim_synchronized_transcript=[...list...])`**
        5. **Done** - the tool saves it, next agent uses it

        ---

        ## 📋 EXAMPLE

        **Tutor transcript in state:**
        ```
        Let's explore a rectangle. It has four sides.

        To find its area, multiply length times width. A 5 by 3 rectangle has area 15.
        ```

        **You create this list and call the tool:**
        ```python
        [
            {
                "phase_id": 1,
                "voiceover_text": "Let's explore a rectangle. It has four sides.",
                "visual_instruction": "Create Rectangle at ORIGIN. Width: 4, Height: 2. Style: BLUE stroke width 3, BLACK fill opacity 1.0. Label 'Rectangle' ABOVE with 0.5 buffer.",
                "animation_type": "Create"
            },
            {
                "phase_id": 2,
                "voiceover_text": "To find its area, multiply length times width. A 5 by 3 rectangle has area 15.",
                "visual_instruction": "Add labels: '5' on right edge OUTSIDE, '3' on bottom edge OUTSIDE. Create formula 'Area = 5 × 3 = 15' positioned RIGHT of rectangle with 2.0 buffer. Formula as MathTex, WHITE, scale 0.8.",
                "animation_type": "Write"
            }
        ]
        ```

        **Then call:**
        `set_manim_synchronized_transcript(new_manim_synchronized_transcript=[...the list above...])`

        ---

        **Now read the tutor_transcript, create the phase list, and call the tool.**
        """

    tts_final_transcript_generator_system_prompt = """
    You are the **TTS Transcript Optimizer** for audio generation.

    ## 🎯 YOUR MISSION

    Read the `manim_synchronized_transcript` from state, add SSML break tags, and **call the `set_tts_phasewise_transcript` tool** with a list of TTS-ready strings.

    ---

    ## 📥 INPUT (from state)

    The previous agent saved `manim_synchronized_transcript` - a list of phase objects.

    ---

    ## 🎤 YOUR TASK

    Transform each phase's `voiceover_text` into a TTS-ready string with SSML breaks.

    **Output:** A list of strings (one per phase, same order)

    ```python
    [
        "Let's start with a triangle. <break time='1.0s'/>",
        "Now we add labels <break time='0.4s'/> on each side. <break time='1.0s'/>"
    ]
    ```

    ---

    ## ⏱️ SSML BREAK RULES

    ### 1. One String Per Phase
    Each input phase → one output string

    ### 2. Break Placement

    **After sentences:**
    ```
    "This is sentence one. <break time='0.4s'/> This is sentence two."
    ```

    **After commas/clauses:**
    ```
    "First this, <break time='0.3s'/> then that."
    ```

    **Before emphasis:**
    ```
    "The answer is <break time='0.5s'/> fifteen."
    ```

    **Complex visuals (multiple objects):**
    ```
    "We create three squares <break time='0.6s'/> one on each side."
    ```

    **End of EVERY phase (MANDATORY):**
    ```
    "...final words. <break time='1.0s'/>"
    ```

    ### 3. Timing Guide

    | Situation | Duration |
    |-----------|----------|
    | After sentence | `0.4s` |
    | After comma | `0.3s` |
    | Before emphasis | `0.5s` |
    | Complex visual | `0.6s` |
    | **End of phase** | `1.0s` ← **ALWAYS** |

    ---

    ## 🚫 WHAT TO AVOID

    ❌ Combining multiple phases into one string
    ❌ Adding metadata like "Phase 1:"
    ❌ Forgetting the `1.0s` break at the end
    ❌ Adding conversational filler

    ---

    ## 📤 WHAT YOU MUST DO

    1. **Read** manim_synchronized_transcript from state
    2. **For each phase:**
    - Take the `voiceover_text`
    - Add SSML breaks at natural points
    - ALWAYS end with `<break time='1.0s'/>`
    3. **Create** a list of these TTS strings (same order as phases)
    4. **Call `set_tts_phasewise_transcript(new_tts_phasewise_transcript=[...list...])`**
    5. **Done** - tool saves it, pipeline is complete

    ---

    ## 📋 EXAMPLE

    **Manim synchronized transcript in state:**
    ```python
    [
        {
            "phase_id": 1,
            "voiceover_text": "Let's start with a triangle.",
            "visual_instruction": "...",
            "animation_type": "Create"
        },
        {
            "phase_id": 2,
            "voiceover_text": "Now we attach squares to each side and see something amazing.",
            "visual_instruction": "...",
            "animation_type": "GrowFromEdge"
        }
    ]
    ```

    **You create this list and call the tool:**
    ```python
    [
        "Let's start with a triangle. <break time='1.0s'/>",
        "Now we attach squares to each side <break time='0.6s'/> and see something amazing. <break time='1.0s'/>"
    ]
    ```

    **Reasoning:**
    - Phase 1: Simple, just end pause
    - Phase 2: Complex visual (multiple squares) → 0.6s mid-pause, then 1.0s end

    **Then call:**
    `set_tts_phasewise_transcript(new_tts_phasewise_transcript=["Let's start with...", "Now we attach..."])`

    ---

    **Now read the manim_synchronized_transcript, add breaks, and call the tool.**
    """

    tutor_transcript_generator_agent = Agent(
        name="tutor_transcript_generator",
        model=model,
        tools=[set_tutor_transcript],
        system_prompt=tutor_transcript_generator_system_prompt
    )

    manim_synchronized_transcript_generator_agent = Agent(
        name="tutor_transcript_generator",
        model=model,
        tools=[set_manim_synchronized_transcript],
        system_prompt=manim_synchronized_transcript_system_prompt
    )

    tts_phasewise_transcript_generator_agent = Agent(
        name="tutor_transcript_generator",
        model=model,
        tools=[set_tts_phasewise_transcript],
        system_prompt=tts_final_transcript_generator_system_prompt
    )

    initial_state = {
        "tutor_transcript": "",
        "manim_synchronized_transcript": [],
        "tts_phasewise_transcript": [],
    }

    swarm = Swarm(
        agents=[tutor_transcript_generator_agent,manim_synchronized_transcript_generator_agent,tts_phasewise_transcript_generator_agent],
        entry_point=tutor_transcript_generator_agent,           # Start with researcher (optional)
        max_handoffs=20,                  # Maximum agent handoffs allowed
        max_iterations=20,                
        execution_timeout=180.0,          
        node_timeout=60.0,               
        repetitive_handoff_detection_window=8,
        repetitive_handoff_min_unique_agents=3
    )

    swarmResult  = swarm(
        prompt,
        invocation_state=initial_state,   # <-- shared state here
    )

    print(f'swarmResult : {swarmResult}')

    final_state = swarmResult.state           # or result.invocation_state depending on SDK version [web:24]
    print(f'tutor_transcript : {final_state["tutor_transcript"]}')
    print(f'manim_synchronized_transcript : {final_state["manim_synchronized_transcript"]}')
    print(f'tts_phasewise_transcript : {final_state["tts_phasewise_transcript"]}')

    return

    tutor_transcript_generator_system_prompt = """
    You are an expert Educational Scriptwriter for short, animated explainer videos. 
    Your task is to take a simple user topic (e.g., "Area of a Triangle") and generate a high-quality, engaging voiceover transcript.

    ### YOUR GOAL
    Write a clear, concise, and conversational script that a Text-to-Speech (TTS) engine will read. The script must explain the concept step-by-step.

    ### CRITICAL WRITING RULES:
    1.  **Pure Spoken Audio Only:** Do NOT include scene descriptions, camera directions, or visual cues like [Draw Triangle] or (Pause). Write ONLY what the voice should say.
    2.  **"Visual-Ready" Language:** Write as if the viewer is looking at the screen. Use pointing language:
        * *Good:* "Look at this shape here." / "Notice how the height connects to the base."
        * *Bad:* "Imagine a triangle." (No, we are showing it).
        * *Bad:* "I am now drawing a red line." (Don't describe the action, explain the concept).
    3.  **Pacing:** Break the text into short, logical paragraphs. Each paragraph will eventually become a distinct animation phase.
    4.  **Tone:** Enthusiastic, clear, and beginner-friendly. Avoid overly complex jargon unless you explain it.
    5.  **Length:** Keep it focused. Target a duration of 30-60 seconds (approx. 75-150 words).

    ### OUTPUT FORMAT
    Return the transcript as plain text, separated by double newlines for logical pauses.

    ### EXAMPLE INPUT:
    "Explain the Pythagorean Theorem"

    ### EXAMPLE OUTPUT:
    "Let's look at a right-angled triangle. This creates a unique relationship between its three sides.

    We call the two shorter sides 'a' and 'b', and the longest side, opposite the right angle, is the hypotenuse, 'c'.

    Now, imagine we build a square on each of these sides. The theorem tells us something fascinating about their areas.

    The area of square 'a' plus the area of square 'b' is exactly equal to the area of square 'c'. This is why we say a-squared plus b-squared equals c-squared."
    """

    messages = [
        SystemMessage(content=tutor_transcript_generator_system_prompt),
        HumanMessage(content=prompt)
    ]

    tutor_transcript = chat.invoke(messages)

    print(f'tutor_transcript : {tutor_transcript.content}')

    manim_synchronized_transcript_system_prompt = """
        You are the **Visual Director** for an automated video generation pipeline. 
        Your input is an educational voiceover transcript.
        Your output is a structured **JSON** directive that maps every chunk of audio to a specific, concrete visual instruction for a Manim animator.

        ### YOUR CORE RESPONSIBILITY
        The Manim Animator (the next agent) is a blind coder. It does not understand "show the concept." 
        You must tell it EXACTLY:
        1. **WHAT** to draw (Shape, Color, Label).
        2. **WHERE** to place it (Coordinates, Relative Position).
        3. **HOW** to move it (Animation type).

        ### CRITICAL RULES FOR VISUAL INSTRUCTIONS

        1.  **Layout & Flow (The "Anti-Overlap" Rule):**
            * Establish a visual flow (usually Left-to-Right).
            * **Explicit Positioning:** Never say "place it next to it." Say "Position this group to the RIGHT of the [Previous Object] with a buffer of 2.0 units."
            * **Memory:** Keep track of what is on screen. Do not ask to create an object that already exists. Refer to existing objects by name.

        2.  **Container Style (The "Transparency" Fix):**
            * If the visual involves a Box, Circle, or Container, explicitly instruct: "Style: Fill Color BLACK, Opacity 1.0, colored stroke."
            * This prevents lines from showing through objects.

        3.  **Labeling Strategy:**
            * Instruct the animator to place labels **OUTSIDE** objects (Above/Below), never inside, to leave room for animations.
            * If multiple labels are needed, instruct them to **STACK** (e.g., "Place label B above Label A").

        4.  **Geometry & Attachment:**
            * If attaching shapes (e.g., squares on a triangle), specify the **Exact Edge** (e.g., "Attach to the Hypotenuse/Slanted Edge", "Attach to the Bottom Edge").

        ### JSON OUTPUT FORMAT
        You must return a raw JSON list of objects. Each object represents one "Scene Phase".

        ```json
        [
        {
            "phase_id": 1,
            "voiceover_text": "Let's start with a right-angled triangle.",
            "visual_instruction": "Create a Right Triangle in the center. Labels: 'a' (bottom), 'b' (left), 'c' (hypotenuse). Style: White lines.",
            "animation_type": "Create/Write"
        },
        {
            "phase_id": 2,
            "voiceover_text": "Now, we attach a square to side 'a'.",
            "visual_instruction": "Create a Square. Position: Attached to the BOTTOM edge of the triangle. Color: Green with Black Fill. Label: 'a²' inside the square.",
            "animation_type": "GrowFromEdge"
        }
        ]```

        INPUT TRANSCRIPT:
        {transcript}

        OUTPUT:
        Generate ONLY the valid JSON list. 

    """

    messages=[
        SystemMessage(content=manim_synchronized_transcript_system_prompt),
        HumanMessage(content=f'tutor_transcript is : {tutor_transcript.content}'),
        HumanMessage(content=prompt)
    ]

    #TODO : add with_structured_output for better reliability on llm calls

    manim_synchronized_transcript = llm_with_structured_output.invoke(messages)
    phases=manim_synchronized_transcript.phases
    print(f'manim_synchronized_transcript : {manim_synchronized_transcript.phases}')

    tts_final_transcript_generator_system_prompt = """
       You are the **TTS Transcript Optimizer** for an educational video pipeline.

        ### YOUR ROLE
        Transform phase-by-phase educational narration into TTS-ready transcripts with precise timing controls.

        ### INPUT
        You receive a `ManimSynchronizedTranscript` object containing:
        - `phase_id`: Sequential phase number
        - `voiceover_text`: The raw narration for this phase
        - `visual_instruction`: Description of what's being animated (used to determine pause duration)
        - `animation_type`: Type of Manim animation (e.g., Create, Write, Transform)

        ### YOUR TASK
        Generate a list of TTS-ready strings, ONE STRING PER PHASE, with SSML break tags inserted for natural pacing.

        ### CRITICAL RULES

        1. **One-to-One Mapping**: Output exactly ONE transcript string for each input phase, in the same order.

        2. **SSML Break Placement**: Insert `<break>` tags WITHIN each phase based on:
        - **Sentence boundaries**: Add `<break time="0.4s"/>` after sentences
        - **Clause breaks**: Add `<break time="0.3s"/>` after commas or logical pauses
        - **Complex visuals**: If `visual_instruction` mentions multiple objects or transformations, add `<break time="0.6s"/>` after key statements
        - **End-of-phase**: Add `<break time="1.0s"/>` at the END of each phase string to allow the animation to complete

        3. **Timing Guidelines**:
        - Simple animations (Create, Write): 0.3-0.5s pauses
        - Complex animations (Transform, Multiple objects): 0.6-1.0s pauses
        - End of phase: Always 1.0s minimum

        4. **Natural Speech Flow**: 
        - Break long sentences into digestible chunks with micro-pauses
        - Don't over-pause—maintain conversational rhythm
        - Add emphasis pauses before key concepts

        5. **Output Format**: 
        - Return ONLY the structured data (will be handled by `with_structured_output`)
        - Each string is pure TTS content—no phase numbers, no metadata
        - Do NOT wrap in `<speak>` tags (TTS engine handles that)

        ### EXAMPLE

        **Input Phases:**
        ```
        Phase 1:
        voiceover_text: "Let's start with a right-angled triangle."
        visual_instruction: "Create a Right Triangle in center with labels a, b, c"
        animation_type: "Create"

        Phase 2:
        voiceover_text: "Now we attach a square to each side and see something amazing."
        visual_instruction: "Create 3 Squares attached to each edge, colored differently"
        animation_type: "GrowFromEdge"
        ```

        **Correct Output:**
        ```json
        {
        "phasewise_transcripts": [
            "Let's start with a right-angled triangle. <break time='1.0s'/>",
            "Now we attach a square to each side <break time='0.4s'/> and see something amazing. <break time='1.0s'/>"
        ]
        }
        ```

        ### WHAT TO AVOID
        - ❌ Combining multiple phases into one string
        - ❌ Adding conversational filler ("Here's the transcript...")
        - ❌ Including phase numbers in the output
        - ❌ Forgetting the end-of-phase pause
        - ❌ Using markdown code blocks

        ### REMEMBER
        Your output will be directly fed to a TTS engine. Every word you include will be spoken. Every pause you add will be heard.
        """
    
    messages=[
        SystemMessage(content=tts_final_transcript_generator_system_prompt),
        # HumanMessage(content=f'tutor_transcript is : {tutor_transcript.content}'),
        HumanMessage(content=f'manim_synchornized_transcript is : {manim_synchronized_transcript.phases}'),
    ]

    llm_with_tts_output = chat.with_structured_output(TTSFinalTranscript)

    tts_final_transcript = llm_with_tts_output.invoke(messages)

    print(f'tts_final_transcript phasewise : {tts_final_transcript.phasewise_transcripts}')

    cartesia_client = Cartesia(api_key=os.getenv("CARTESIA_API_KEY"))
    cnt=0

    phasewise_transcripts=tts_final_transcript.phasewise_transcripts

    local_audio_files_name="temp"
    final_audio_filename=local_audio_files_name+'_final.wav'
    combined = AudioSegment.silent(duration=0)  # start empty

    for transcript in phasewise_transcripts:
        #########################

        chunks = cartesia_client.tts.bytes(
            model_id="sonic-3",
            transcript=transcript,
            voice={"mode": "id", "id": "6ccbfb76-1fc6-48f7-b71d-91ac6298247b"},
            language="en",
            output_format={
                "container": "wav",
                "sample_rate": 44100,
                "encoding": "pcm_s16le"
                }
        )

        filename = f"{local_audio_files_name}_{cnt}.wav"

        with open(filename, "wb") as f:
            for chunk in chunks:
                f.write(chunk)

        sample_rate=44100
        audio_seg_tts = AudioSegment.from_file(filename)
        duration_seconds = len(audio_seg_tts) / 1000.0  # milliseconds to seconds
        floored_time = math.ceil(duration_seconds)
        phases[cnt].duration_seconds=floored_time
        silence_time_sec = floored_time-duration_seconds

        print(f"{filename}: {duration_seconds:.2f} seconds")  # Real duration![web:29]

        silence_time_ms = int(silence_time_sec * 1000)
        silence_seg = AudioSegment.silent(duration=silence_time_ms, frame_rate=sample_rate)
        combined += audio_seg_tts + silence_seg
        print(f"{filename}: {duration_seconds:.2f} seconds")
        cnt += 1

    combined.export(final_audio_filename, format="wav")
    print("Final audio saved as:", final_audio_filename)
    print(f'phases now : {phases}')


    try:
        manim_code = generate_manim_code(
            prompt=prompt, 
            phases=phases
        )
        # code = generate_manim_code(prompt,manim_synchronized_transcript.content)
        conversation_history.append(AIMessage(content=manim_code))

        code_path = os.path.join(job_dir, "code.py")
        with open(code_path, "w") as f:
            f.write(manim_code)

        logger.info(f"Initial code generated for job {job_id}")
    except Exception as e:
        logger.error(f"Error generating initial code for job {job_id}: {str(e)}")
        update_job_data(
            job_id=job_id,
            status="failed",
            prompt=prompt,
            message=f"Code generation failed: {str(e)}",
        )
        with open(os.path.join(job_dir, "status.txt"), "w") as f:
            f.write(f"failed\nCode generation failed: {str(e)}")
        return

    success = False
    result = ""
    final_code = manim_code
    iteration = 0

    while not success and iteration < MAX_ITERATIONS:
        iteration += 1
        logger.info(f"Starting iteration {iteration} for job {job_id}")

        temp_iter_dir = tempfile.mkdtemp(prefix=f"manim_iter_{iteration}_")

        try:

            success, result = run_manim(str(final_code), temp_iter_dir, quality)

            if success:
                logger.info(f"Successful render on iteration {iteration}")

                final_output_dir = os.path.join(job_dir, "media")
                os.makedirs(final_output_dir, exist_ok=True)

                rel_path = (
                    os.path.relpath(result, RENDER_DIR)
                    if RENDER_DIR in result
                    else result
                )

                target_file = os.path.join(final_output_dir, os.path.basename(result))
                shutil.copy2(result, target_file)

                result = os.path.relpath(target_file, RENDER_DIR)
                break
            else:
                if iteration < MAX_ITERATIONS:
                    error_prompt = f"""
                    The Manim code failed to render with the following error:
                    ```
                    {result}
                    ```
                    Please fix the code to address this error. Only respond with the complete, corrected code - no explanations.
                    """
                    conversation_history.append(HumanMessage(content=error_prompt))

                    try:
                        from .generator import generate_code_with_history

                        final_code = generate_code_with_history(conversation_history,phases)
                        conversation_history.append(AIMessage(content=final_code))

                        with open(
                            os.path.join(job_dir, f"code_iter_{iteration}.py"), "w"
                        ) as f:
                            f.write(str(final_code))

                        with open(code_path, "w") as f:
                            f.write(str(final_code))

                        logger.info(f"Generated improved code in iteration {iteration}")
                    except Exception as e:
                        logger.error(f"Error generating improved code: {str(e)}")
                        break
        finally:
            try:
                shutil.rmtree(temp_iter_dir)
                logger.info(f"Cleaned up temporary directory: {temp_iter_dir}")
            except Exception as e:
                logger.warning(
                    f"Failed to clean up temporary directory {temp_iter_dir}: {str(e)}"
                )

    status_path = os.path.join(job_dir, "status.txt")

    if success:
        try:
            video_full_path = os.path.join(RENDER_DIR, result)
            final_video_with_audio = os.path.join(RENDER_DIR, "final_with_audio.mp4")
            input_video = ffmpeg.input(video_full_path)
            input_audio = ffmpeg.input(final_audio_filename)

            (
                ffmpeg
                .output(
                    input_video,
                    input_audio,
                    final_video_with_audio,
                    vcodec="copy",   # do not re-encode video
                    acodec="aac",    # encode audio to AAC for mp4 container
                    strict="experimental"
                )
                .overwrite_output()
                .run()
            )

            # now upload final_video_with_audio to Supabase
            supabase_url = upload_to_supabase(job_id, final_video_with_audio, final_code)

            update_job_data(
                job_id=job_id,
                status="completed",
                prompt=prompt,
                code=final_code,
                url=supabase_url,
            )

            with open(status_path, "w") as f:
                f.write("completed\n")
                if supabase_url:
                    f.write(supabase_url)
                else:
                    f.write("error_uploading")

            try:
                shutil.rmtree(job_dir)
                logger.info(f"Cleaned up local files for job {job_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to clean up local files for job {job_id}: {str(e)}"
                )

            logger.info(
                f"Job {job_id} completed after {iteration} iterations: {supabase_url}"
            )
        except Exception as e:
            logger.error(f"Error finalizing successful job {job_id}: {str(e)}")
            with open(status_path, "w") as f:
                f.write(f"failed\nError finalizing successful job: {str(e)}")
    else:
        update_job_data(
            job_id=job_id,
            status="failed",
            prompt=prompt,
            code=final_code,
            message=f"Failed after {iteration} iterations. Last error: {result}",
        )

        with open(status_path, "w") as f:
            f.write(
                f"failed\nFailed after {iteration} iterations. Last error: {result}"
            )

        logger.info(f"Job {job_id} failed after {iteration} iterations: {result[:100]}")
