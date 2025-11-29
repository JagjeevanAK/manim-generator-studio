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


def process_rendering_job(job_id: str, prompt: str, quality: str):
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

    manim_synchronized_transcript = chat.invoke(messages)

    print(f'manim_synchronized_transcript : {manim_synchronized_transcript}')

    tts_final_transcript_generator_system_prompt = tts_final_transcript_generator_system_prompt = """
        You are the **Audio Mastering Agent**.
        Your ONLY goal is to output a raw string for a Text-to-Speech (TTS) engine.

        ### INPUT DATA
        You will receive a JSON list of video phases. Each phase contains:
        - `voiceover_text`: The spoken words.
        - `visual_instruction`: The context for the pause.

        ### YOUR TASK
        Convert the list into a single, continuous string with SSML break tags inserted between phrases.

        ### STRICT OUTPUT RULES
        1.  **NO MARKDOWN:** Do not wrap the output in ```xml or ```txt.
        2.  **NO JSON:** Do not output a key-value pair.
        3.  **NO CONVERSATION:** Do not say "Here is the text." Just output the text itself.
        4.  **SSML TAGS:** Use `<break time="2.0s" />` for standard pauses and `<break time="3.0s" />` for complex visual transitions.

        ### LOGIC
        1. Read Phase 1 `voiceover_text`.
        2. Append a break tag based on Phase 1 `visual_instruction` complexity.
        3. Read Phase 2 `voiceover_text`.
        4. Append break tag...
        5. Repeat until done.

        ### EXAMPLE INTERACTION

        **Input:**
        [
        { "voiceover_text": "First, we draw a circle.", "visual_instruction": "Create Circle" },
        { "voiceover_text": "Then, we fill it with blue.", "visual_instruction": "Animate Fill" }
        ]

        **Correct Output:**
        First, we draw a circle. <break time="2.0s" /> Then, we fill it with blue. <break time="2.0s" />
    """

    messages=[
        SystemMessage(content=tts_final_transcript_generator_system_prompt),
        HumanMessage(content=f'tutor_transcript is : {tutor_transcript.content}'),
        HumanMessage(content=f'manim_synchornized_transcript is : {manim_synchronized_transcript.content}'),
    ]

    tts_final_transcript = chat.invoke(messages)

    print(f'tts_final_transcript : {tts_final_transcript.content}')

    

    try:
        code = generate_manim_code(prompt)
        conversation_history.append(AIMessage(content=code))

        code_path = os.path.join(job_dir, "code.py")
        with open(code_path, "w") as f:
            f.write(code)

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
    final_code = code
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

                        final_code = generate_code_with_history(conversation_history)
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
            supabase_url = upload_to_supabase(job_id, video_full_path, final_code)

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
