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
from strands.models.litellm import LiteLLMModel

model = LiteLLMModel(
    client_args={
        "api_key":settings.OPENROUTER_API_KEY,
    },
    # model_id="openrouter/openai/gpt-4o",
    model_id="openrouter/openai/gpt-4.5",
    # model_id="openrouter/google/gemini-2.0-flash-lite-001",
    # model_id="openrouter/google/gemini-2.0-flash-exp:free",
    # model_id="openrouter/google/gemini-2.0-flash-001",
    # model_id="openrouter/google/gemini-2.5-pro",
    params={
        'temperature':0.5,
        "max_tokens":2000
    },
)

# model = LiteLLMModel(
#     client_args={
#         "api_key": settings.COHERE_API_KEY,  # Use your Cohere API key here
#     },
#     model_id="cohere_chat/command-a-03-2025",  # Direct Cohere model ID for LiteLLM
#     params={
#         'temperature': 0.5,
#         "max_tokens": 1000
#     },
# )


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
    code_pattern = re.compile(r"(?:python)?\s*([\s\S]*?)\s*")
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

      
        #updated for aws strands agent
        manim_code_generator_system_prompt = """
            You are an expert Manim animation developer. Generate a single, complete `GeneratedScene(Scene)` class that synchronizes perfectly with pre-rendered audio.
            IMP : return FINAL code ONLY direct python ! not in ```python code```

            ## INPUT STRUCTURE
            You receive `TranscriptPhase` objects with:
            - `phase_id`: Sequential number
            - `voiceover_text`: Narrator's speech (for context)
            - `visual_instruction`: What to animate
            - `animation_type`: Suggested animation (Create, Write, Transform, etc.)
            - `duration_seconds`: EXACT audio duration (critical!)

            ## TIMING RULES (MOST IMPORTANT)
            Each phase MUST match its exact `duration_seconds`:
                ANIMATION_TIME = duration_seconds * 0.7  # 70% animate
            PAUSE_TIME = duration_seconds * 0.3      # 30% settle

            # Example: duration=6.0s
            self.play(Create(rect), run_time=4.2)  # 6.0 * 0.7
            self.wait(1.8)  # 6.0 * 0.3
            IMP : return FINAL code ONLY direct python ! not in ```python code```
            

            **Special cases:**
            - `NoChange`: Use `self.wait(duration_seconds)` only
            - Multiple animations: Split 70% time equally, then add 30% pause
            - Never hardcode wait times - always calculate from duration

            ## VISUAL BEST PRACTICES
            1. **Solid containers** (prevent transparency artifacts):
                rect = Rectangle(fill_color=BLACK, fill_opacity=1.0, 
                            stroke_color=WHITE, stroke_width=3)
            

            2. **Labels outside objects** (avoid overlap):
                label = Text("Title").scale(0.6)
            label.next_to(box, UP, buff=0.3)
            

            3. **Edge-based connections**:
                arrow = Arrow(box1.get_right(), box2.get_left(), buff=0.1)
            

            4. **Group related objects**:
                diagram = VGroup(shape, label1, label2)
            

            ## ANIMATION MAPPING
            - Create → `Create()`
            - Write → `Write()`
            - Transform → `Transform(obj1, obj2)`
            - FadeIn/Out → `FadeIn()`, `FadeOut()`
            - Indicate → `Indicate()`
            - NoChange → `self.wait(duration)`

            ## OUTPUT FORMAT
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
            
            IMP : return FINAL code ONLY direct python ! not in ```python code```

            ## REQUIREMENTS
            - Return ONLY executable Python code (no markdown, no explanations)
            - Comment each phase with voiceover text and duration
            - Use the `rag` tool to fetch Manim documentation when needed
            - Generate complete, crash-free code
            - All timings MUST sum exactly to duration_seconds per phase

            Generate the scene based on the phases provided.

            VERY VERY IMP : 
            i. Only output final python code directly not even python
            ii.Cross check at the end whether generated code has any overlapping part
            OVERLAPPING parts damage the user experience and they are FORBIDDEN!


            DO NOT USE LATEX ! no matter what 

            Do not assume any constants exist. Always define positions, coordinates, or objects before using them

            IMP : return FINAL code ! not in ```python ```
            """

        #compresses for aws strands agent
        manim_code_generator_system_prompt = """
You are an expert Manim animation developer. Generate a single, complete Python scene class `GeneratedScene(Scene)` that perfectly synchronizes with the provided audio.  

## INPUT
- `phases`: List of TranscriptPhase objects with:
  - `phase_id`: Sequential number
  - `voiceover_text`: Narrator speech (for context)
  - `visual_instruction`: What to animate
  - `animation_type`: Suggested animation (Create, Write, Transform, etc.)
  - `duration_seconds`: Exact audio duration (critical)

## TIMING RULES
- Each phase must match its `duration_seconds`:
    - ANIMATION_TIME = duration_seconds * 0.7
    - PAUSE_TIME = duration_seconds * 0.3
- Special cases:
    - `NoChange` → `self.wait(duration_seconds)`
    - Multiple animations → split 70% time equally, then 30% pause
- Never hardcode wait times; always calculate from duration

## VISUAL BEST PRACTICES
- Use solid rectangles for containers
- Place labels outside objects to avoid overlap
- Use edge-based arrows for connections
- Group related objects using `VGroup`

## ANIMATION MAPPING
- Create → `Create()`
- Write → `Write()`
- Transform → `Transform(obj1, obj2)`
- FadeIn/Out → `FadeIn()`, `FadeOut()`
- Indicate → `Indicate()`
- NoChange → `self.wait(duration)`

## OUTPUT REQUIREMENTS
- Return ONLY executable Python code (no markdown, no explanations, no LaTeX)
- Comment each phase with `voiceover_text` and `duration_seconds`
- Ensure no overlapping animations; overlapping is forbidden
- Define all positions, coordinates, and objects before use
- Generate complete, crash-free code
- Cross-check timings and object creation

Generate the scene now. Only output **final Python code**.
"""


        rewrite_manim_code_system_prompt = """
        You are a Manim code debugging specialist. Your task is to analyze a failed Manim script and generate corrected, fully executable Python code that preserves timing and visual instructions.  

        ## INPUT
        - `phases`: List of TranscriptPhase objects with `voiceover_text`, `visual_instruction`, `animation_type`, and `duration_seconds`.
        - `recent_manim_code`: Code that failed to compile.
        - `recent_error`: Exact error message.

        ## TASK
        1. Identify the root cause from the error.
        2. Correct the code while preserving timing synchronization:
        - Animation time = duration_seconds * 0.7
        - Pause time = duration_seconds * 0.3
        3. Ensure all objects exist before use.
        4. Follow phase instructions exactly.
        5. Comment only what was fixed.

        ## COMMON ERROR FIXES
        - Missing imports → `from manim import *` (+ `import numpy as np` if needed)
        - Wrong Scene class → `class Scene(Scene):`
        - Animation syntax → Create objects first, then animate
        - Undefined variables → Always define objects before animating
        - Transform syntax → Transform(Mobject1, Mobject2)
        - Invalid geometry → Use valid coordinates (Polygon needs ≥3 points)

        ## VISUAL BEST PRACTICES
        - Use solid rectangles for containers
        - Place labels outside objects
        - Use edge-based arrows
        - Group related objects with VGroup

        ## OUTPUT REQUIREMENTS
        - Return ONLY executable Python code (no markdown, no explanations, no LaTeX)
        - Ensure no overlapping animations; overlapping is forbidden
        - Do not assume constants exist; define all positions and objects
        - Cross-check timings and object creation

        Generate the corrected code now. Only output **final Python code**.
        """

        agent = Agent(
            model=model,
            tools=[rag],
            system_prompt=manim_code_generator_system_prompt
        )


        agent = Agent(
            model=model,
            tools=[], #temporary testing with cohere ai (cohere agent dont support tools in aws strands)
            system_prompt=manim_code_generator_system_prompt
        )

        response = agent(f'Prefetch manim documentations : {context_text} , phases :{phases}')
        print(f'\nresponse from code gen agent aws strands\n{response}\n')

        manim_code = response.message["content"][0]["text"]  # {"role": "assistant", "content": [ ... ]}

        print(f'manim_code : {manim_code}')
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


def generate_code_with_history(errors,phases:list[any],recent_manim_code:str):
    """
    Generate improved Manim code using conversation history and error feedback.

    Args:
        conversation_history: List of HumanMessage and AIMessage instances

    Returns:
        str: Generated Python code
    """
    try:
        # original_prompt = error_history[0].content

        doc_search = PineconeVectorStore(
            index_name=settings.PINECONE_INDEX_NAME,
            embedding=embedding,
            pinecone_api_key=settings.PINECONE_API_KEY,
        )

        retriever = doc_search.as_retriever()
        docs = retriever.invoke(errors[-1])

        doc_contents = [doc.page_content for doc in docs[:5]]
        context_text = "\n\n---\n\n".join(doc_contents)

     
        #updated system prompt for aws strands agent
        rewrite_manim_code_system_prompt = """
            You are a Manim debugging specialist. Analyze the failed code and error, then generate CORRECTED code that fixes all issues while maintaining perfect audio synchronization.
            IMP : return FINAL code ONLY direct python ! not in ```python code```

            ## INPUT DATA
            - `phases`: List of `TranscriptPhase` objects with timing/visual requirements
            - `recent_manim_code`: The code that failed compilation
            - `recent_error`: The exact error message

            ## YOUR TASK
            1. Identify the root cause from the error message
            2. Fix the issue while preserving timing synchronization
            3. Return corrected, executable code

            ## COMMON ERROR PATTERNS & FIXES

                        IMP : return FINAL code ONLY direct python ! not in ```python code```

            ### 1. Missing Imports
            **Error:** `NameError: name 'Circle' is not defined`
            **Fix:** Add `from manim import *` (and `import numpy as np` if using np.array)

            ### 2. Wrong Class Name
            **Error:** `TypeError: Scene.construct() takes 1 positional argument`
            **Fix:** Use exactly `class Scene(Scene):` (not MyScene, VideoScene, etc.)

            ### 3. Animation Syntax
            **Error:** `TypeError: GrowArrow() takes 2 positional arguments`
            **Fix:**
                # ❌ Wrong: self.play(GrowArrow(start, end))
            # ✅ Correct:
            arrow = Arrow(start, end)
            self.play(GrowArrow(arrow))
            

            ### 4. Undefined Variables
            **Error:** `NameError: name 'triangle' is not defined`
            **Fix:** Create objects BEFORE animating them:
                # ✅ Correct order:
            triangle = Polygon([-2,-1,0], [2,-1,0], [0,2,0])
            self.play(Create(triangle))  # Now triangle exists
            

            ### 5. Transform Syntax
            **Error:** `Transform() requires Mobjects`
            **Fix:**
                # ❌ Wrong: Transform("A", "B")
            # ✅ Correct:
            text1, text2 = Text("A"), Text("B")
            self.play(Transform(text1, text2))
            
            IMP : return FINAL code ONLY direct python ! not in ```python code```

            ### 6. Invalid Geometry
            **Error:** `ValueError: Polygon needs at least 3 points`
            **Fix:** Provide valid 3D coordinates:
                triangle = Polygon(
                [-2, -1, 0],  # 3D point format
                [2, -1, 0],
                [0, 2, 0]
            )
            

            ## TIMING SYNCHRONIZATION (CRITICAL)
            Each phase MUST match its `duration_seconds` exactly:
                ANIMATION_TIME = duration_seconds * 0.7
            PAUSE_TIME = duration_seconds * 0.3

            # Example: duration=6.0s
            self.play(Create(obj), run_time=4.2)  # 6.0 * 0.7
            self.wait(1.8)  # 6.0 * 0.3

            IMP : return FINAL code ONLY direct python ! not in ```python code```
            

            **Special cases:**
            - `NoChange`: `self.wait(duration_seconds)` only
            - Multiple animations: Split 70% time, add 30% pause at end
            - Never hardcode wait times

            ## VISUAL BEST PRACTICES
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
            

            ## OUTPUT FORMAT
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
            

            ## REQUIREMENTS
            - Return ONLY executable Python code (no markdown, no explanations)
            - Fix the specific error identified
            - Maintain exact timing from phases
            - Comment what was fixed
            - Use `rag` tool for Manim documentation if needed
            - Ensure all objects exist before use
            - Calculate all timings from duration_seconds

            Analyze the error and generate corrected code now.

            VERY VERY IMP :
            i. Only output final python code directly not even python
            ii.Cross check at the end whether generated code has any overlapping part
            OVERLAPPING parts damage the user experience and they are FORBIDDEN!
            
            DO NOT USE LATEX ! no matter what 
            Do not assume any constants exist. Always define positions, coordinates, or objects before using them

            IMP : return FINAL code ONLY direct python ! not in ```python code```
            """

        #compresses for aws strands agent
        rewrite_manim_code_system_prompt = """
        You are a Manim code debugging specialist. Your task is to analyze a failed Manim script and generate corrected, fully executable Python code that preserves timing and visual instructions.  

        ## INPUT
        - `phases`: List of TranscriptPhase objects with `voiceover_text`, `visual_instruction`, `animation_type`, and `duration_seconds`.
        - `recent_manim_code`: Code that failed to compile.
        - `recent_error`: Exact error message.

        ## TASK
        1. Identify the root cause from the error.
        2. Correct the code while preserving timing synchronization:
        - Animation time = duration_seconds * 0.7
        - Pause time = duration_seconds * 0.3
        3. Ensure all objects exist before use.
        4. Follow phase instructions exactly.
        5. Comment only what was fixed.

        ## COMMON ERROR FIXES
        - Missing imports → `from manim import *` (+ `import numpy as np` if needed)
        - Wrong Scene class → `class Scene(Scene):`
        - Animation syntax → Create objects first, then animate
        - Undefined variables → Always define objects before animating
        - Transform syntax → Transform(Mobject1, Mobject2)
        - Invalid geometry → Use valid coordinates (Polygon needs ≥3 points)

        ## VISUAL BEST PRACTICES
        - Use solid rectangles for containers
        - Place labels outside objects
        - Use edge-based arrows
        - Group related objects with VGroup

        ## OUTPUT REQUIREMENTS
        - Return ONLY executable Python code (no markdown, no explanations, no LaTeX)
        - Ensure no overlapping animations; overlapping is forbidden
        - Do not assume constants exist; define all positions and objects
        - Cross-check timings and object creation

        Generate the corrected code now. Only output **final Python code**.
        """


        agent = Agent(
            model=model,
            tools=[rag],
            system_prompt=rewrite_manim_code_system_prompt
        )    

        #temporary testing with cohere ai agent with aws strands
        agent = Agent(
            model=model,
            tools=[], #temporary testing with cohere ai (cohere agent dont support tools in aws strands)
            system_prompt=rewrite_manim_code_system_prompt
        )    
        
        response = agent(f'phases : {phases},  recent error : {errors[-1]}, recent manim code :{recent_manim_code}, previous errros : {errors}')
        print(f'\noutput from rewrite manim code agent\n{response}\n')
        rewritren_manim_code=response.message["content"][0]["text"]

        print(f'rewritten manim code by aws strands agent\n{rewritren_manim_code}')
        return recent_manim_code
    
        messages=[
            SystemMessage(content=system_prompt),
            # HumanMessage(content=f'### MANIM DOCUMENTATION CONTEXT (for syntax reference): : {context_text}'),
            HumanMessage(content=f"### PHASES: : {phases}"),
            HumanMessage(content=f"### Manim documentation context: : {context_text}"),
            HumanMessage(content=f"### Conversation history : {conversation_history},Rewrite the code now :")
        ]

        response = cohere.invoke(messages)

        logger.info("Generated improved code with error context")
        print('------------------Improved Manim code : ')
        print(response.content)
        code = extract_python_code(response.content)
        return code

    except Exception as e:
        logger.error(f"Error generating improved code: {str(e)}")


        return recent_manim_code
        # last_code = None
        # for msg in reversed(conversation_history):
        #     if isinstance(msg, AIMessage):
        #         last_code = msg.content
        #         break

        # if last_code:
        #     return last_code
        # else:
        #     return f"# Error generating improved code: {str(e)}"
