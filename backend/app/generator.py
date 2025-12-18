import re
import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere, CohereEmbeddings
from .config import settings
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_pinecone import PineconeVectorStore
from strands import Agent,tool

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# embedding = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
#     google_api_key=settings.GEMINI_API_KEY,
# )

embedding = CohereEmbeddings(
    model="embed-multilingual-v2.0",
    cohere_api_key=settings.COHERE_API_KEY
)

# chat = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     verbose=True,
#     google_api_key=settings.GEMINI_API_KEY
# )

chat = ChatCohere(
    model="command-r-plus-08-2024", 
    verbose=True,
    cohere_api_key=settings.COHERE_API_KEY,
    temperature=0.3 
)

cohere = ChatCohere(
    model="command-a-03-2025", 
    verbose=True,
    cohere_api_key=settings.COHERE_API_KEY,
    temperature=0.3 
)

chat_history = []

MAX_HISTORY_LENGTH = 20


def add_message_to_history(message):
    """Add a message to the global chat history and trim if needed."""
    global chat_history
    chat_history.append(message)
    if len(chat_history) > MAX_HISTORY_LENGTH:
        chat_history = chat_history[-MAX_HISTORY_LENGTH:]


def extract_python_code(text: str) -> str:
    """Extract Python code from text that might contain markdown code blocks."""
    code_pattern = re.compile(r"```(?:python)?\s*([\s\S]*?)\s*```")
    matches = code_pattern.findall(text)
    return matches[0] if matches else text


# @tool
def rag(queries:str):
    """Rag tool on manim documentations

    Args: 
        queries: str with all the queries for RAG
    """
    doc_search = PineconeVectorStore(
        index_name=settings.PINECONE_INDEX_NAME,
        embedding=embedding,
        pinecone_api_key=settings.PINECONE_API_KEY,
    )

    retriever = doc_search.as_retriever()

    docs_original = retriever.invoke()

    docs_enhanced = retriever.invoke(queries)

    all_docs = docs_original + docs_enhanced
    unique_docs = []
    seen_content = set()

    for doc in all_docs:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            unique_docs.append(doc)

    doc_contents = []
    for doc in unique_docs[:7]:
        doc_contents.append(doc.page_content)

    return unique_docs

def generate_manim_code(prompt: str,phases:list[any]) -> str:
    """Generate Manim code using Cohere model with improved retrieval context."""
    add_message_to_history(HumanMessage(content=prompt))

    query_enhancement_prompt = """
        You are a Manim Library Search Optimizer. 
        Your task is to generate a precise search query for the Manim documentation based on the User's Topic AND the specific Visual Plan.

        ### INPUTS
        1. **User Topic:** {input}
        2. **Visual Plan (JSON):** {manim_synchronized_transcript}

        ### YOUR TASK
        Construct a search query by analyzing both inputs:

        1.  **Analyze the Visual Plan:** Look at the `visual_instruction` fields in the JSON.
            * If it mentions "arrows", add `Arrow`, `GrowArrow`.
            * If it mentions "grids" or "tiles", add `VGroup`, `Square`, `arrange_in_grid`.
            * If it mentions "braces" or "labels", add `Brace`, `Text`, `next_to`.
            * If it mentions "transforming", add `ReplacementTransform`, `Transform`.

        2.  **Identify the Mathematical Domain (from User Topic):**
            * Calculus -> `Axes`, `Graph`, `TangentLine`.
            * Geometry -> `Polygon`, `Angle`, `dashed_line`.
            * Linear Algebra -> `Vector`, `Matrix`, `LinearTransformationScene`.

        3.  **Include Layout Keywords:**
            * Always include: `VGroup`, `arrange`, `next_to`, `align_to`.

        ### OUTPUT FORMAT
        Output ONLY a comma-separated list of the top 5-10 most relevant search terms. Do not add explanations.

        Enhanced search query:"""

    try:
        messages=[
            SystemMessage(content=query_enhancement_prompt),
            HumanMessage(content=f'phases : {phases}')
        ]

        enhanced_query_response = chat.invoke(
           messages
        )

        enhanced_query = enhanced_query_response.content
        logger.info(f"Enhanced search query: {enhanced_query}")

        doc_search = PineconeVectorStore(
            index_name=settings.PINECONE_INDEX_NAME,
            embedding=embedding,
            pinecone_api_key=settings.PINECONE_API_KEY,
        )

        retriever = doc_search.as_retriever()

        docs_original = retriever.invoke(prompt)

        docs_enhanced = retriever.invoke(enhanced_query)

        all_docs = docs_original + docs_enhanced
        unique_docs = []
        seen_content = set()

        for doc in all_docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_docs.append(doc)

        doc_contents = []
        for doc in unique_docs[:7]:
            doc_contents.append(doc.page_content)

        context_text = f"\n\n---\n\n".join(doc_contents)

        logger.info(f"Retrieved {len(unique_docs)} unique relevant documents")

        manim_code_generator_system_prompt = """
        You are an **Elite Audio-Synchronized Manim Developer**.

        Your mission: Generate a single, complete Manim `Scene` class that perfectly synchronizes with pre-rendered audio phases.

        ---

        ## 🎯 CRITICAL CONTEXT: AUDIO-FIRST ARCHITECTURE

        **THE AUDIO IS ALREADY RENDERED.** Each phase has a corresponding audio file with an EXACT duration in seconds.
        Your generated animations MUST match these durations PRECISELY. No guessing, no approximations.

        ---

        ## 📥 INPUT DATA STRUCTURE

        You will receive a list of `TranscriptPhase` objects with these fields:

        ```python
        TranscriptPhase(
            phase_id=1,
            voiceover_text="Let's explore the area of a rectangle...",
            visual_instruction="Create a Rectangle in center. Label: 'Rectangle'. Style: Dashed lines.",
            animation_type="Create",
            duration_seconds=6.234  # ← EXACT audio duration for this phase
        )
        ```

        ### Key Fields:
        - **`phase_id`**: Sequential phase number (1, 2, 3...)
        - **`voiceover_text`**: What the narrator is saying (for context, not code)
        - **`visual_instruction`**: EXACT visual requirements for this phase
        - **`animation_type`**: Suggested Manim animation (Create, Write, Transform, etc.)
        - **`duration_seconds`**: **CRITICAL** - The EXACT time this audio lasts

        ---

        ## ⏱️ TIMING SYNCHRONIZATION RULES (MOST IMPORTANT)

        ### Phase Timing Formula:
        ```
        TOTAL_PHASE_TIME = duration_seconds
        ANIMATION_TIME = duration_seconds * 0.7  (70% for animation)
        PAUSE_TIME = duration_seconds * 0.3       (30% for settling/comprehension)
        ```

        ### Implementation Pattern:
        ```python
        # Phase 1: duration_seconds = 6.0
        self.play(
            Create(rectangle),
            run_time=4.2  # 6.0 * 0.7 = 4.2s for animation
        )
        self.wait(1.8)  # 6.0 * 0.3 = 1.8s pause

        # Phase 2: duration_seconds = 10.5
        self.play(
            Write(label),
            run_time=7.35  # 10.5 * 0.7
        )
        self.wait(3.15)  # 10.5 * 0.3
        ```

        ### Special Cases:
        1. **NoChange Animation** (e.g., transitional voiceover):
        - Use `self.wait(duration_seconds)` for the entire phase
        - No visual changes

        2. **Multiple Animations in One Phase**:
        - Distribute the 70% animation time across all animations
        - Example: If phase is 12s with 3 animations:
            ```python
            anim_time = 12 * 0.7 / 3  # 2.8s per animation
            self.play(Create(obj1), run_time=2.8)
            self.play(Write(label), run_time=2.8)
            self.play(Indicate(obj1), run_time=2.8)
            self.wait(12 * 0.3)  # 3.6s pause
            ```

        3. **Complex Transformations**:
        - If `visual_instruction` involves multiple objects, use `AnimationGroup` or `LaggedStart`
        - Still respect the total phase duration

        ---

        ## 🎨 VISUAL IMPLEMENTATION RULES

        ### 1. Container Styling (The "Transparency Fix")
        ```python
        # ✅ CORRECT: Solid containers that block background
        rectangle = Rectangle(
            width=4, height=2,
            fill_color=BLACK,      # Solid black fill
            fill_opacity=1.0,      # Fully opaque
            stroke_color=WHITE,    # Visible outline
            stroke_width=3
        )

        # ❌ WRONG: Transparent containers show artifacts
        rectangle = Rectangle(fill_opacity=0)  # Lines will show through!
        ```

        ### 2. Label Placement (The "Anti-Collision Strategy")
        ```python
        # ✅ CORRECT: Labels OUTSIDE objects
        title = Text("Box A").scale(0.6)
        title.next_to(box, UP, buff=0.3)

        # If adding a second label above the first:
        value = Text("Value: 42").scale(0.5)
        value.next_to(title, UP, buff=0.2)  # Stack on the TITLE, not the box

        # ❌ WRONG: Labels inside can get obscured
        title.move_to(box.get_center())  # Will overlap with content
        ```

        ### 3. Spatial Layout (The "Flow System")
        ```python
        # ✅ CORRECT: Left-to-right flow with explicit positioning
        group1 = VGroup(shape1, label1)
        group2 = VGroup(shape2, label2)
        group2.next_to(group1, RIGHT, buff=2.0)  # Clear separation

        # ❌ WRONG: Overlapping placements
        group2.move_to(group1.get_center())  # Will overlap!
        ```

        ### 4. Edge-Based Connections
        ```python
        # ✅ CORRECT: Arrows connect to edges
        arrow = Arrow(
            box1.get_right(),      # From right edge
            box2.get_left(),       # To left edge
            buff=0.1
        )

        # ❌ WRONG: Arrows to centers cross through objects
        arrow = Arrow(box1.get_center(), box2.get_center())
        ```

        ---

        ## 🔧 ANIMATION TYPE MAPPING

        Interpret `animation_type` from the phases:

        | animation_type | Manim Class | Usage |
        |----------------|-------------|-------|
        | `Create` | `Create()` | Draw shapes, lines |
        | `Write` | `Write()` | Text, equations |
        | `Transform` | `Transform()` | Morph object A → B |
        | `GrowFromEdge` | `GrowFromEdge(edge=DOWN)` | Expand from edge |
        | `FadeIn` | `FadeIn()` | Soft appearance |
        | `FadeOut` | `FadeOut()` | Soft disappearance |
        | `Indicate` | `Indicate()` | Highlight/emphasize |
        | `Flash` | `Flash()` | Attention burst |
        | `ShowCreation` | `Create()` | Legacy alias |
        | `NoChange` | `self.wait()` | Just pause |

        ### Handling Visual Instructions:

        **Pattern Recognition:**
        - **"Grid" / "Tiles"** → Nested loops with `VGroup`
        - **"Fill" / "Color"** → `set_fill()` with animation
        - **"Highlight"** → `Indicate()` or `Circumscribe()`
        - **"Dashed line"** → `DashedLine()`
        - **"Attach to edge"** → Use `.next_to(obj, direction)` or `.align_to()`

        ---

        ## 🛡️ SYNTAX SAFETY RULES

        ### 1. Arrow Creation
        ```python
        # ✅ CORRECT
        self.play(GrowArrow(Arrow(start, end)))

        # ❌ WRONG
        self.play(GrowArrow(start, end))  # GrowArrow needs an Arrow object
        ```

        ### 2. Transform Requirements
        ```python
        # ✅ CORRECT
        text1 = Text("A")
        text2 = Text("B")
        self.play(Transform(text1, text2))

        # ❌ WRONG
        self.play(Transform("A", "B"))  # Needs Mobjects, not strings
        ```

        ### 3. Path Animations
        ```python
        # ✅ CORRECT
        path = Line(start, end)
        self.play(MoveAlongPath(dot, path))

        # ❌ WRONG
        self.play(MoveAlongPath(dot, start, end))  # Needs a path object
        ```

        ### 4. VGroup Usage
        ```python
        # ✅ CORRECT: Group related objects for unified control
        diagram = VGroup(triangle, label_a, label_b, label_c)
        diagram.move_to(ORIGIN)

        # Later, can animate the entire group
        self.play(FadeOut(diagram))
        ```

        ---

        ## 📋 CODE STRUCTURE TEMPLATE

        ```python
        from manim import *

        class GeneratedScene(Scene):
            def construct(self):
                # Phase 1: [voiceover_text here for reference]
                # Duration: {duration_seconds}s
                # Visual: {visual_instruction}
                
                {objects_creation}
                self.play(
                    {animation_type}({objects}),
                    run_time={duration_seconds * 0.7}
                )
                self.wait({duration_seconds * 0.3})
                
                # Phase 2: [voiceover_text]
                # Duration: {duration_seconds}s
                ...
                
                # Continue for all phases
        ```

        ---

        ## 🎬 COMPLETE EXAMPLE

        **Input Phases:**
        ```python
        [
            TranscriptPhase(
                phase_id=1,
                voiceover_text="Let's start with a triangle.",
                visual_instruction="Create a Triangle in center. Label: 'Triangle'.",
                animation_type="Create",
                duration_seconds=4.5
            ),
            TranscriptPhase(
                phase_id=2,
                voiceover_text="Now we label the sides a, b, and c.",
                visual_instruction="Add labels 'a', 'b', 'c' to each side.",
                animation_type="Write",
                duration_seconds=6.0
            )
        ]
        ```

        **Expected Output:**
        ```python
        from manim import *

        class GeneratedScene(Scene):
            def construct(self):
                # Phase 1: "Let's start with a triangle."
                # Duration: 4.5s
                triangle = Polygon(
                    [-2, -1, 0], [2, -1, 0], [0, 2, 0],
                    fill_color=BLACK,
                    fill_opacity=1.0,
                    stroke_color=WHITE,
                    stroke_width=3
                )
                title = Text("Triangle").scale(0.7)
                title.next_to(triangle, UP, buff=0.5)
                
                tri_group = VGroup(triangle, title)
                
                self.play(
                    Create(triangle),
                    run_time=3.15  # 4.5 * 0.7
                )
                self.play(Write(title), run_time=0)  # Quick follow-up
                self.wait(1.35)  # 4.5 * 0.3
                
                # Phase 2: "Now we label the sides a, b, and c."
                # Duration: 6.0s
                label_a = MathTex("a").scale(0.8)
                label_a.next_to(triangle, DOWN, buff=0.3)
                
                label_b = MathTex("b").scale(0.8)
                label_b.next_to(triangle, LEFT, buff=0.3)
                
                label_c = MathTex("c").scale(0.8)
                label_c.next_to(triangle, RIGHT, buff=0.3)
                
                labels = VGroup(label_a, label_b, label_c)
                
                self.play(
                    Write(labels),
                    run_time=4.2  # 6.0 * 0.7
                )
                self.wait(1.8)  # 6.0 * 0.3
        ```

        ---

        ## 🚨 CRITICAL REMINDERS

        1. **NEVER hardcode `self.wait(2)`** - ALWAYS calculate from `duration_seconds`
        2. **TOTAL phase time = animation time + wait time MUST equal `duration_seconds`**
        3. **Use the 70/30 split** unless visual_instruction explicitly requires different timing
        4. **For NoChange phases**, use `self.wait(duration_seconds)` for the entire duration
        5. **Comment each phase** with its voiceover text and duration for debugging
        6. **Group related objects** in VGroups for easier management
        7. **Test edge cases**: What if duration is 1.5s? Still apply the formula (1.05s animation, 0.45s wait)

        ---

        ## 📦 OUTPUT REQUIREMENTS

        Return ONLY executable Python code:
        - Start with `from manim import *`
        - Define a single `class GeneratedScene(Scene):`
        - Include phase comments for debugging
        - NO explanatory text outside the code
        - NO markdown formatting
        - NO placeholders - generate complete, working code

        ---

        ## 🎯 YOUR TASK

        Given the phases below, generate a complete, crash-free Manim scene that synchronizes perfectly with the audio.


        Use tool : 'rag' for retreiving documentation of manim codes for your code genration process whenever necessary

        """

        #updated for aws strands agent
        manim_code_generator_system_prompt = """
            You are an expert Manim animation developer. Generate a single, complete `GeneratedScene(Scene)` class that synchronizes perfectly with pre-rendered audio.

            ## INPUT STRUCTURE
            You receive `TranscriptPhase` objects with:
            - `phase_id`: Sequential number
            - `voiceover_text`: Narrator's speech (for context)
            - `visual_instruction`: What to animate
            - `animation_type`: Suggested animation (Create, Write, Transform, etc.)
            - `duration_seconds`: EXACT audio duration (critical!)

            ## TIMING RULES (MOST IMPORTANT)
            Each phase MUST match its exact `duration_seconds`:
            ```python
            ANIMATION_TIME = duration_seconds * 0.7  # 70% animate
            PAUSE_TIME = duration_seconds * 0.3      # 30% settle

            # Example: duration=6.0s
            self.play(Create(rect), run_time=4.2)  # 6.0 * 0.7
            self.wait(1.8)  # 6.0 * 0.3
            ```

            **Special cases:**
            - `NoChange`: Use `self.wait(duration_seconds)` only
            - Multiple animations: Split 70% time equally, then add 30% pause
            - Never hardcode wait times - always calculate from duration

            ## VISUAL BEST PRACTICES
            1. **Solid containers** (prevent transparency artifacts):
            ```python
            rect = Rectangle(fill_color=BLACK, fill_opacity=1.0, 
                            stroke_color=WHITE, stroke_width=3)
            ```

            2. **Labels outside objects** (avoid overlap):
            ```python
            label = Text("Title").scale(0.6)
            label.next_to(box, UP, buff=0.3)
            ```

            3. **Edge-based connections**:
            ```python
            arrow = Arrow(box1.get_right(), box2.get_left(), buff=0.1)
            ```

            4. **Group related objects**:
            ```python
            diagram = VGroup(shape, label1, label2)
            ```

            ## ANIMATION MAPPING
            - Create → `Create()`
            - Write → `Write()`
            - Transform → `Transform(obj1, obj2)`
            - FadeIn/Out → `FadeIn()`, `FadeOut()`
            - Indicate → `Indicate()`
            - NoChange → `self.wait(duration)`

            ## OUTPUT FORMAT
            ```python
            from manim import *

            class GeneratedScene(Scene):
                def construct(self):
                    # Phase 1: [voiceover_text]
                    # Duration: {duration}s
                    
                    {create_objects}
                    self.play(
                        {animation}({objects}),
                        run_time={duration * 0.7}
                    )
                    self.wait({duration * 0.3})
                    
                    # Phase 2...
            ```

            ## REQUIREMENTS
            - Return ONLY executable Python code (no markdown, no explanations)
            - Comment each phase with voiceover text and duration
            - Use the `rag` tool to fetch Manim documentation when needed
            - Generate complete, crash-free code
            - All timings MUST sum exactly to duration_seconds per phase

            Generate the scene based on the phases provided.
            """

        agent = Agent(
            model=model,
            tools=[rag],
            system_prompt=manim_code_generator_system_prompt
        )

        manim_code = agent(f'Prefetch manim documentations : {context_text} , phases :{phases}')

        print(f'maim code genrated by aws starnds agent\n{manim_code}')
        return manim_code
    
        messages=[
            SystemMessage(content=manim_code_generator_system_prompt),
            HumanMessage(content=f'### Prefetched MANIM DOCUMENTATION CONTEXT (for syntax reference): : {context_text}'),
            HumanMessage(content=f"### PHASES: : {phases},\nGenerate the code now :")
        ]

        response = cohere.invoke(messages)

        # logger.info(f"Generated response for prompt: {prompt[:30]}...")

        code = extract_python_code(response.content)
        print('-----------------------GENARATED manim code -----------------------')
        print(code)
        return code

    except Exception as e:
        logger.error(f"Error generating code: {str(e)}")
        return f"# Error generating code: {str(e)}"


def generate_code_with_history(conversation_history,phases:list[any],recent_manim_code:str):
    """
    Generate improved Manim code using conversation history and error feedback.

    Args:
        conversation_history: List of HumanMessage and AIMessage instances

    Returns:
        str: Generated Python code
    """
    try:
        original_prompt = conversation_history[0].content

        doc_search = PineconeVectorStore(
            index_name=settings.PINECONE_INDEX_NAME,
            embedding=embedding,
            pinecone_api_key=settings.PINECONE_API_KEY,
        )

        retriever = doc_search.as_retriever()
        docs = retriever.invoke(original_prompt)

        doc_contents = [doc.page_content for doc in docs[:5]]
        context_text = "\n\n---\n\n".join(doc_contents)

        rewrite_manim_code_system_prompt="""
            ERROR CORRECTION & AUDIO-SYNCHRONIZED MANIM CODE GENERATOR
            =============================================================

            You are an **Elite Manim Debugging & Audio Synchronization Specialist**.

            Your mission: Analyze the conversation history (original request, previous code attempts, errors), 
            and generate CORRECTED, audio-synchronized Manim code that:
            1. ✅ Fixes all previous errors
            2. ✅ Synchronizes perfectly with pre-rendered audio phases
            3. ✅ Produces a crash-free, working animation

            ---

            ## 🎯 CRITICAL CONTEXT: YOU'RE FIXING BROKEN CODE

            **CONVERSATION HISTORY STRUCTURE:**
            - **First message**: Original user request (e.g., "Explain area of triangle")
            - **Subsequent messages**: Previous code attempts and their error messages
            - **Current task**: Generate code that addresses ALL previous failures

            **Common Error Patterns to Watch For:**
            1. Missing imports (`NameError: name 'Circle' is not defined`)
            2. Wrong class names (must be `Scene`, not `VideoScene` or `MyScene`)
            3. Incorrect animation syntax (e.g., `GrowArrow(start, end)` instead of `GrowArrow(Arrow(start, end))`)
            4. Timing issues (hardcoded waits that don't match audio)
            5. Object reference errors (using variables before creation)
            6. Geometry errors (triangles with invalid coordinates)

            ---

            ## ⏱️ AUDIO SYNCHRONIZATION REQUIREMENTS

            **THE AUDIO IS ALREADY RENDERED.** You will receive a list of `TranscriptPhase` objects:

            ```python
            TranscriptPhase(
                phase_id=1,
                voiceover_text="Let's explore the area of a rectangle...",
                visual_instruction="Create a Rectangle in center. Label: 'Rectangle'.",
                animation_type="Create",
                duration_seconds=6.234  # ← EXACT audio duration - MUST match exactly
            )
            ```

            ### MANDATORY TIMING FORMULA:
            ```
            For EACH phase:
                ANIMATION_TIME = duration_seconds × 0.7  (70% for visual animations)
                PAUSE_TIME = duration_seconds × 0.3      (30% for comprehension/settling)
                
                TOTAL_PHASE_TIME = ANIMATION_TIME + PAUSE_TIME = duration_seconds
            ```

            ### Implementation Pattern:
            ```python
            # Phase 1: duration_seconds = 6.0
            self.play(
                Create(rectangle),
                run_time=4.2  # 6.0 × 0.7 = 4.2s
            )
            self.wait(1.8)    # 6.0 × 0.3 = 1.8s

            # Phase 2: duration_seconds = 10.5  
            self.play(
                Write(label),
                run_time=7.35  # 10.5 × 0.7
            )
            self.wait(3.15)    # 10.5 × 0.3
            ```

            ### Special Timing Cases:

            **1. NoChange Animation (Transitional voiceover with no visual change):**
            ```python
            # Phase has animation_type='NoChange', duration_seconds=5.0
            # Just wait for the entire duration
            self.wait(5.0)
            ```

            **2. Multiple Animations in One Phase:**
            ```python
            # Phase has 3 animations, total duration = 12s
            anim_time_each = 12 × 0.7 / 3  # 2.8s per animation

            self.play(Create(obj1), run_time=2.8)
            self.play(Write(label), run_time=2.8)  
            self.play(Indicate(obj1), run_time=2.8)
            self.wait(12 × 0.3)  # 3.6s pause at end
            ```

            **3. Simultaneous Animations (using AnimationGroup):**
            ```python
            # Phase needs multiple things happening together, duration = 8s
            self.play(
                AnimationGroup(
                    Create(shape1),
                    Create(shape2),
                    Write(label)
                ),
                run_time=5.6  # 8 × 0.7
            )
            self.wait(2.4)  # 8 × 0.3
            ```

            ---

            ## 🛠️ ERROR CORRECTION STRATEGIES

            ### 1. Missing Imports Analysis
            **Look for errors like:**
            ```
            NameError: name 'Circle' is not defined
            AttributeError: module 'manim' has no attribute 'Polygon'
            ```

            **Fix:** Add proper imports at the top:
            ```python
            from manim import *
            import numpy as np  # Only if using np.array, np.sin, etc.
            ```

            ### 2. Class Name Issues
            **Error pattern:**
            ```
            TypeError: Scene.construct() takes 1 positional argument but 2 were given
            ```

            **Fix:** ALWAYS use exactly this class signature:
            ```python
            class Scene(Scene):  # ✅ CORRECT
                def construct(self):
                    ...

            # ❌ WRONG variations:
            class MyScene(Scene):  # Wrong name
            class VideoScene(Scene):  # Wrong name  
            class Scene():  # Missing parent class
            ```

            ### 3. Animation Syntax Errors
            **Common mistakes from conversation history:**

            ```python
            # ❌ WRONG: GrowArrow needs an Arrow object
            self.play(GrowArrow(start_point, end_point))

            # ✅ CORRECT:
            arrow = Arrow(start_point, end_point)
            self.play(GrowArrow(arrow))

            # ❌ WRONG: Transform needs Mobjects, not strings
            self.play(Transform("A", "B"))

            # ✅ CORRECT:
            text1 = Text("A")
            text2 = Text("B")
            self.play(Transform(text1, text2))

            # ❌ WRONG: Using undefined variables
            self.play(Create(triangle))  # triangle was never created

            # ✅ CORRECT:
            triangle = Polygon([0,0,0], [2,0,0], [1,2,0])
            self.play(Create(triangle))
            ```

            ### 4. Geometry & Coordinate Errors
            **If previous attempt had:**
            ```
            ValueError: Polygon needs at least 3 points
            IndexError: list index out of range
            ```

            **Fix with proper coordinates:**
            ```python
            # ✅ CORRECT: Valid triangle
            triangle = Polygon(
                [-2, -1, 0],  # Bottom left
                [2, -1, 0],   # Bottom right  
                [0, 2, 0],    # Top
                stroke_color=WHITE,
                stroke_width=3
            )

            # ✅ CORRECT: Proper rectangle
            rect = Rectangle(
                width=4,
                height=2,
                fill_color=BLACK,
                fill_opacity=1.0,
                stroke_color=WHITE
            )
            ```

            ### 5. Object Reference & Ordering Errors
            **Check conversation history for:**
            ```
            NameError: name 'label_a' is not defined
            AttributeError: 'NoneType' object has no attribute 'get_center'
            ```

            **Fix:** Create objects BEFORE using them:
            ```python
            # ✅ CORRECT order:
            triangle = Polygon(...)  # Create first
            self.play(Create(triangle))  # Then animate
            label = Text("A").next_to(triangle, UP)  # Then reference it
            self.play(Write(label))

            # ❌ WRONG order:
            label = Text("A").next_to(triangle, UP)  # Error! triangle doesn't exist yet
            triangle = Polygon(...)
            ```

            ---

            ## 🎨 VISUAL IMPLEMENTATION RULES

            ### 1. Container Styling (Prevents Visual Artifacts)
            ```python
            # ✅ CORRECT: Solid, opaque containers
            box = Rectangle(
                width=3, height=2,
                fill_color=BLACK,      # Solid background
                fill_opacity=1.0,      # Fully opaque
                stroke_color=BLUE,     # Visible border
                stroke_width=3
            )

            # ❌ WRONG: Transparent containers show underlying objects
            box = Rectangle(fill_opacity=0)  # Lines/arrows show through!
            ```

            ### 2. Label Placement Strategy
            ```python
            # ✅ CORRECT: Labels OUTSIDE objects to avoid collision
            title = Text("Box A").scale(0.6)
            title.next_to(box, UP, buff=0.3)

            # If stacking multiple labels:
            subtitle = Text("Value: 42").scale(0.5)
            subtitle.next_to(title, UP, buff=0.2)  # Stack on TITLE, not box

            # ❌ WRONG: Labels inside get obscured
            title.move_to(box.get_center())  # Overlaps with box content
            ```

            ### 3. Spatial Layout & Flow
            ```python
            # ✅ CORRECT: Left-to-right flow with clear separation
            obj1_group = VGroup(shape1, label1)
            obj2_group = VGroup(shape2, label2)

            obj2_group.next_to(obj1_group, RIGHT, buff=2.0)  # Clear spacing

            # ❌ WRONG: Overlapping positions
            obj2_group.move_to(obj1_group.get_center())  # Will overlap!
            ```

            ### 4. Arrow Connections
            ```python
            # ✅ CORRECT: Connect arrows to edges, not centers
            arrow = Arrow(
                box1.get_right(),   # From right edge of box1
                box2.get_left(),    # To left edge of box2
                buff=0.1
            )

            # ❌ WRONG: Arrows through centers cross objects
            arrow = Arrow(box1.get_center(), box2.get_center())
            ```

            ---

            ## 🔧 ANIMATION TYPE MAPPING

            Interpret `animation_type` from phases:

            | animation_type | Manim Implementation | Use Case |
            |----------------|---------------------|----------|
            | `Create` | `Create(obj)` | Draw shapes, lines, geometric objects |
            | `Write` | `Write(obj)` | Text, labels, equations |
            | `FadeIn` | `FadeIn(obj)` | Soft appearance |
            | `FadeOut` | `FadeOut(obj)` | Soft disappearance |
            | `Transform` | `Transform(obj1, obj2)` | Morph object into another |
            | `GrowFromEdge` | `GrowFromEdge(obj, edge=DOWN)` | Grow from specific edge |
            | `Indicate` | `Indicate(obj)` | Highlight/emphasize existing object |
            | `Flash` | `Flash(obj)` | Attention burst effect |
            | `Circumscribe` | `Circumscribe(obj)` | Draw attention with surrounding shape |
            | `ShowCreation` | `Create(obj)` | Legacy name, use Create |
            | `NoChange` | `self.wait(duration_seconds)` | No visual change, just pause |

            ### Visual Instruction Patterns:

            **Pattern Recognition & Implementation:**

            ```python
            # "Create a grid of 3x3 squares"
            squares = VGroup(*[
                Square(side_length=0.5).shift(RIGHT*i + UP*j)
                for i in range(3) for j in range(3)
            ])

            # "Fill the shape with blue color"
            self.play(shape.animate.set_fill(BLUE, opacity=0.7), run_time=...)

            # "Highlight the formula"
            self.play(Indicate(formula), run_time=...)

            # "Draw a dashed line from A to B"
            dashed = DashedLine(point_a, point_b)

            # "Attach square to the bottom edge of triangle"
            square = Square(side_length=2)
            square.next_to(triangle, DOWN, buff=0)
            # OR align to specific edge:
            square.align_to(triangle.get_bottom(), UP)
            ```

            ---

            ## 📋 CODE GENERATION TEMPLATE

            ```python
            from manim import *

            class Scene(Scene):  # ← MUST be exactly "Scene"
                def construct(self):
                    # Phase 1: [voiceover_text for context]
                    # Duration: {duration_seconds}s
                    # Visual: {visual_instruction}
                    # Fix from previous error: [specific fix applied]
                    
                    {create_objects}
                    
                    self.play(
                        {animation}({objects}),
                        run_time={duration_seconds * 0.7}
                    )
                    self.wait({duration_seconds * 0.3})
                    
                    # Phase 2: ...
                    # Continue for all phases
            ```

            ---

            ## 🎬 COMPLETE ERROR-CORRECTION EXAMPLE

            **Conversation History Shows:**
            ```
            Attempt 1 Error: NameError: name 'Triangle' is not defined
            Attempt 2 Error: Wrong class name 'MyScene'
            Attempt 3 Error: Timing too short, audio cuts off
            ```

            **Input Phases:**
            ```python
            [
                TranscriptPhase(
                    phase_id=1,
                    voiceover_text="Let's look at a triangle.",
                    visual_instruction="Create a Triangle with vertices at [-2,-1,0], [2,-1,0], [0,2,0]",
                    animation_type="Create",
                    duration_seconds=4.5
                )
            ]
            ```

            **Corrected Output:**
            ```python
            from manim import *  # ← Fix: Added missing import

            class Scene(Scene):  # ← Fix: Correct class name (was MyScene)
                def construct(self):
                    # Phase 1: "Let's look at a triangle."
                    # Duration: 4.5s
                    # Fixed: Proper Polygon syntax, correct timing
                    
                    triangle = Polygon(
                        [-2, -1, 0],
                        [2, -1, 0],
                        [0, 2, 0],
                        fill_color=BLACK,
                        fill_opacity=1.0,
                        stroke_color=WHITE,
                        stroke_width=3
                    )
                    
                    self.play(
                        Create(triangle),
                        run_time=3.15  # ← Fix: 4.5 × 0.7 (was hardcoded 2s)
                    )
                    self.wait(1.35)  # ← Fix: 4.5 × 0.3 (was missing)
            ```

            ---

            ## 🚨 CRITICAL CHECKLIST BEFORE GENERATING CODE

            Review conversation history and ensure:

            - [ ] **Imports**: `from manim import *` at the top
            - [ ] **Class name**: Exactly `class Scene(Scene):`
            - [ ] **All objects created** before being referenced
            - [ ] **Animation syntax** matches Manim documentation
            - [ ] **Timing formula applied** to EVERY phase: `run_time = duration × 0.7`, `wait = duration × 0.3`
            - [ ] **NoChange phases** just use `self.wait(duration_seconds)`
            - [ ] **Previous errors addressed** with comments explaining the fix
            - [ ] **Coordinates are valid** (3D points for Polygon, proper dimensions for shapes)
            - [ ] **VGroups used** for related objects
            - [ ] **Labels placed outside** objects to avoid collision
            - [ ] **No hardcoded waits** - all timing calculated from `duration_seconds`

            ---

            ## 📦 OUTPUT REQUIREMENTS

            1. **Return ONLY executable Python code**
            2. **Start with imports**: `from manim import *`
            3. **Use class name**: `class Scene(Scene):`
            4. **Include phase comments** showing:
            - Phase number
            - Voiceover text (for context)
            - Duration
            - Any fixes applied from previous errors
            5. **NO explanations outside code**
            6. **NO markdown formatting** (no ```python)
            7. **NO placeholders** - complete, working code only
            8. **Calculate all timings** from `duration_seconds` (no hardcoded values)

            ---

            ## 🎯 YOUR TASK

            **ANALYZE the conversation history below:**
            - Identify what the user originally wanted
            - Review all previous code attempts
            - Note every error that occurred
            - Understand what went wrong

            **THEN GENERATE corrected code that:**
            1. Fixes ALL previous errors
            2. Synchronizes perfectly with the audio phases
            3. Follows all visual and timing rules above
            4. Includes comments explaining major fixes


            **Generate the corrected, audio-synchronized code now.**
            """

        #updated system prompt for aws strands agent
        rewrite_manim_code_system_prompt = """
            You are a Manim debugging specialist. Analyze the failed code and error, then generate CORRECTED code that fixes all issues while maintaining perfect audio synchronization.

            ## INPUT DATA
            - `phases`: List of `TranscriptPhase` objects with timing/visual requirements
            - `recent_manim_code`: The code that failed compilation
            - `recent_error`: The exact error message

            ## YOUR TASK
            1. Identify the root cause from the error message
            2. Fix the issue while preserving timing synchronization
            3. Return corrected, executable code

            ## COMMON ERROR PATTERNS & FIXES

            ### 1. Missing Imports
            **Error:** `NameError: name 'Circle' is not defined`
            **Fix:** Add `from manim import *` (and `import numpy as np` if using np.array)

            ### 2. Wrong Class Name
            **Error:** `TypeError: Scene.construct() takes 1 positional argument`
            **Fix:** Use exactly `class Scene(Scene):` (not MyScene, VideoScene, etc.)

            ### 3. Animation Syntax
            **Error:** `TypeError: GrowArrow() takes 2 positional arguments`
            **Fix:**
            ```python
            # ❌ Wrong: self.play(GrowArrow(start, end))
            # ✅ Correct:
            arrow = Arrow(start, end)
            self.play(GrowArrow(arrow))
            ```

            ### 4. Undefined Variables
            **Error:** `NameError: name 'triangle' is not defined`
            **Fix:** Create objects BEFORE animating them:
            ```python
            # ✅ Correct order:
            triangle = Polygon([-2,-1,0], [2,-1,0], [0,2,0])
            self.play(Create(triangle))  # Now triangle exists
            ```

            ### 5. Transform Syntax
            **Error:** `Transform() requires Mobjects`
            **Fix:**
            ```python
            # ❌ Wrong: Transform("A", "B")
            # ✅ Correct:
            text1, text2 = Text("A"), Text("B")
            self.play(Transform(text1, text2))
            ```

            ### 6. Invalid Geometry
            **Error:** `ValueError: Polygon needs at least 3 points`
            **Fix:** Provide valid 3D coordinates:
            ```python
            triangle = Polygon(
                [-2, -1, 0],  # 3D point format
                [2, -1, 0],
                [0, 2, 0]
            )
            ```

            ## TIMING SYNCHRONIZATION (CRITICAL)
            Each phase MUST match its `duration_seconds` exactly:
            ```python
            ANIMATION_TIME = duration_seconds * 0.7
            PAUSE_TIME = duration_seconds * 0.3

            # Example: duration=6.0s
            self.play(Create(obj), run_time=4.2)  # 6.0 * 0.7
            self.wait(1.8)  # 6.0 * 0.3
            ```

            **Special cases:**
            - `NoChange`: `self.wait(duration_seconds)` only
            - Multiple animations: Split 70% time, add 30% pause at end
            - Never hardcode wait times

            ## VISUAL BEST PRACTICES
            ```python
            # Solid containers (prevent artifacts)
            rect = Rectangle(fill_color=BLACK, fill_opacity=1.0,
                            stroke_color=WHITE, stroke_width=3)

            # Labels outside objects
            label = Text("Title").scale(0.6)
            label.next_to(box, UP, buff=0.3)

            # Edge-based arrows
            arrow = Arrow(box1.get_right(), box2.get_left(), buff=0.1)

            # Group related objects
            diagram = VGroup(shape, label1, label2)
            ```

            ## OUTPUT FORMAT
            ```python
            from manim import *

            class Scene(Scene):
                def construct(self):
                    # Phase 1: [voiceover_text]
                    # Duration: {duration}s
                    # Fix applied: [what was corrected]
                    
                    {create_objects}
                    self.play(
                        {animation}({objects}),
                        run_time={duration * 0.7}
                    )
                    self.wait({duration * 0.3})
                    
                    # Continue for all phases...
            ```

            ## REQUIREMENTS
            - Return ONLY executable Python code (no markdown, no explanations)
            - Fix the specific error identified
            - Maintain exact timing from phases
            - Comment what was fixed
            - Use `rag` tool for Manim documentation if needed
            - Ensure all objects exist before use
            - Calculate all timings from duration_seconds

            Analyze the error and generate corrected code now.
            """

        agent = Agent(
            model=model,
            tools=[rag],
            system_prompt=rewrite_manim_code_system_prompt
        )        

        rewritren_manim_code = agent(f'prefetch manim documentations :{context_text} phases : {phases}, recent error: {conversation_history[-1]}, recent manim code :{recent_manim_code}')

        print(f'rewritten manim code by aws strands agent\n{rewritren_manim_code}')
        return recent_manim_code
    
        messages=[
            SystemMessage(content=system_prompt),
            # HumanMessage(content=f'### MANIM DOCUMENTATION CONTEXT (for syntax reference): : {context_text}'),
            HumanMessage(content=f"### PHASES: : {phases}"),
            HumanMessage(content=f"### Manim documentation context: : {context_text}"),
            HumanMessage(content=f"### Conversation history : {conversation_history},\Rewrite the code now :")
        ]

        response = cohere.invoke(messages)

        logger.info("Generated improved code with error context")
        print('------------------Improved Manim code : ')
        print(response.content)
        code = extract_python_code(response.content)
        return code

    except Exception as e:
        logger.error(f"Error generating improved code: {str(e)}")

        last_code = None
        for msg in reversed(conversation_history):
            if isinstance(msg, AIMessage):
                last_code = msg.content
                break

        if last_code:
            return last_code
        else:
            return f"# Error generating improved code: {str(e)}"
