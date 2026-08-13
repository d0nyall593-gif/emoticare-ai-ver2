import streamlit as st
from PIL import Image
import numpy as np
import torch

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM
)


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="EmotiCare AI",
    page_icon="💙",
    layout="centered"
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "stage": "welcome",
    "first_emotion": None,
    "first_confidence": 0.0,
    "second_emotion": None,
    "second_confidence": 0.0,
    "actual_emotion": None,
    "messages": [],
    "final_feeling": None,
}


for key, value in DEFAULTS.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# EMOTION MODEL
# ============================================================

@st.cache_resource
def load_emotion_model():

    return pipeline(
        "image-classification",
        model="trpakov/vit-face-expression"
    )


# ============================================================
# CHAT MODEL
# ============================================================

@st.cache_resource
def load_chat_model():

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )

    return tokenizer, model


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def get_models():

    emotion_model = load_emotion_model()

    tokenizer, chat_model = load_chat_model()

    return emotion_model, tokenizer, chat_model


# Don't load everything immediately on welcome screen.
# Models are loaded when the user actually starts.


# ============================================================
# EMOTION EMOJIS
# ============================================================

EMOTION_EMOJIS = {

    "happy": "😊",

    "sad": "😔",

    "angry": "😡",

    "fear": "😨",

    "surprise": "😮",

    "disgust": "🤢",

    "neutral": "😐",

}


# ============================================================
# EMOTION ANALYSIS
# ============================================================

def analyze_emotion(
    photo,
    emotion_model
):

    image = Image.open(
        photo
    ).convert("RGB")

    results = emotion_model(
        image
    )

    if not results:

        return None, 0.0

    best = results[0]

    emotion = best["label"].lower()

    confidence = float(
        best["score"]
    ) * 100

    return emotion, confidence


# ============================================================
# REAL AI CHAT RESPONSE
# ============================================================

def generate_response(
    tokenizer,
    model,
    emotion,
    messages
):

    system_prompt = f"""
You are EmotiCare, a friendly and supportive
conversation companion for a school project.

The camera estimated that the user's facial
expression may be "{emotion}".

IMPORTANT:
The camera does NOT know the user's true feelings.
Never claim that the user definitely feels an emotion.

Your job is to:
- listen carefully
- respond naturally to what the user actually says
- ask relevant follow-up questions
- remember what they said earlier
- avoid repeating the same sentence
- keep responses short and conversational
- use simple, friendly language
- never pretend to be a doctor or therapist
- never diagnose mental health conditions
- never say you know exactly how the user feels
- do not mention being a language model unless asked

The user should feel like they are having
a genuine conversation, not answering a questionnaire.

If the user gives a short answer, gently encourage
them to explain more.

Do not force the conversation to stay on the
emotion detected by the camera.
"""

    chat_messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    for message in messages:

        chat_messages.append(
            {
                "role": message["role"],
                "content": message["content"]
            }
        )

    prompt = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    return response


# ============================================================
# WELCOME
# ============================================================

if st.session_state.stage == "welcome":

    st.title("💙 EmotiCare AI")

    st.subheader(
        "A moment to check in with yourself."
    )

    st.write(
        """
        EmotiCare uses facial-expression AI as a starting
        point, then lets you have a real AI-powered
        conversation about what's on your mind.
        """
    )

    st.info(
        """
        📸 Camera

        The camera is only used when you choose to
        take an emotion-check photo.
        """
    )

    st.warning(
        """
        ⚠️ Important

        Facial-expression AI cannot know exactly how
        someone feels. It only estimates what expression
        appears in the photo.

        Your own answer is more important than the AI's
        prediction.
        """
    )

    if st.button(
        "💙 Start Check-in",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.stage = "first_scan"

        st.rerun()


# ============================================================
# FIRST SCAN
# ============================================================

elif st.session_state.stage == "first_scan":

    st.title("📸 First Check")

    st.write(
        "Take a photo so the AI can estimate your facial expression."
    )

    photo = st.camera_input(
        "Take your first photo"
    )

    if photo:

        with st.spinner(
            "Loading the AI models..."
        ):

            emotion_model, tokenizer, chat_model = get_models()

        with st.spinner(
            "Analyzing your expression..."
        ):

            try:

                emotion, confidence = analyze_emotion(
                    photo,
                    emotion_model
                )

                st.session_state.first_emotion = emotion

                st.session_state.first_confidence = confidence

                emoji = EMOTION_EMOJIS.get(
                    emotion,
                    "🙂"
                )

                st.success(
                    f"{emoji} Possible expression: "
                    f"**{emotion.capitalize()}**"
                )

                st.caption(
                    f"AI estimate confidence: {confidence:.1f}%"
                )

                st.write(
                    f"Does **{emotion}** match how you feel?"
                )

                col1, col2 = st.columns(2)

                with col1:

                    if st.button(
                        "✅ Yes",
                        use_container_width=True
                    ):

                        st.session_state.actual_emotion = emotion

                        st.session_state.stage = "conversation"

                        st.rerun()

                with col2:

                    if st.button(
                        "❌ No",
                        use_container_width=True
                    ):

                        st.session_state.stage = "correct_emotion"

                        st.rerun()

            except Exception as error:

                st.error(
                    "The emotion model couldn't analyze the photo."
                )

                st.exception(error)


# ============================================================
# CORRECT AI
# ============================================================

elif st.session_state.stage == "correct_emotion":

    st.title("💭 You know yourself better")

    st.write(
        """
        The camera can make mistakes.
        Tell EmotiCare how you're actually feeling.
        """
    )

    choices = [

        "Happy",

        "Sad",

        "Angry",

        "Worried",

        "Stressed",

        "Tired",

        "Excited",

        "Confused",

        "Neutral",

    ]

    selected = st.selectbox(
        "How are you feeling?",
        choices
    )

    if st.button(
        "Continue 💙",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.actual_emotion = selected.lower()

        st.session_state.stage = "conversation"

        st.rerun()


# ============================================================
# REAL AI CONVERSATION
# ============================================================

elif st.session_state.stage == "conversation":

    st.title("💬 Talk to EmotiCare")

    emotion_model, tokenizer, chat_model = get_models()

    # --------------------------------------------------------
    # First AI message
    # --------------------------------------------------------

    if len(st.session_state.messages) == 0:

        opening = (
            f"I noticed your expression may look "
            f"{st.session_state.actual_emotion}. "
            f"Does something have you feeling that way?"
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": opening
            }
        )

    # --------------------------------------------------------
    # Show conversation
    # --------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

    # --------------------------------------------------------
    # User message
    # --------------------------------------------------------

    user_message = st.chat_input(
        "Tell me what's going on..."
    )

    if user_message:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        with st.spinner(
            "EmotiCare is thinking..."
        ):

            try:

                response = generate_response(
                    tokenizer,
                    chat_model,
                    st.session_state.actual_emotion,
                    st.session_state.messages
                )

            except Exception as error:

                response = (
                    "I'm sorry, I had trouble thinking "
                    "of a response just now."
                )

                st.error(
                    str(error)
                )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        st.rerun()

    # --------------------------------------------------------
    # Finish conversation
    # --------------------------------------------------------

    st.divider()

    st.write(
        "When you're ready to finish the conversation:"
    )

    if st.button(
        "💙 How do I feel now?",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.stage = "second_scan"

        st.rerun()


# ============================================================
# SECOND SCAN
# ============================================================

elif st.session_state.stage == "second_scan":

    st.title("💙 How do you feel now?")

    st.write(
        """
        Before we finish, let's take another photo.

        EmotiCare will compare the two facial-expression
        estimates.
        """
    )

    st.caption(
        "The comparison is only an estimate — your own feelings matter most."
    )

    photo = st.camera_input(
        "Take your final photo"
    )

    if photo:

        emotion_model, tokenizer, chat_model = get_models()

        with st.spinner(
            "Analyzing your expression again..."
        ):

            try:

                emotion, confidence = analyze_emotion(
                    photo,
                    emotion_model
                )

                st.session_state.second_emotion = emotion

                st.session_state.second_confidence = confidence

                st.session_state.stage = "result"

                st.rerun()

            except Exception as error:

                st.error(
                    "The second scan failed."
                )

                st.exception(error)


# ============================================================
# RESULT
# ============================================================

elif st.session_state.stage == "result":

    first = st.session_state.first_emotion

    second = st.session_state.second_emotion

    st.title("📊 Your Check-in")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Before")

        st.write(
            f"{EMOTION_EMOJIS.get(first, '🙂')} "
            f"**{first.capitalize()}**"
        )

        st.caption(
            f"{st.session_state.first_confidence:.1f}%"
        )

    with col2:

        st.subheader("Now")

        st.write(
            f"{EMOTION_EMOJIS.get(second, '🙂')} "
            f"**{second.capitalize()}**"
        )

        st.caption(
            f"{st.session_state.second_confidence:.1f}%"
        )

    st.divider()

    if first != second:

        st.success(
            f"""
            The AI's estimated facial expression changed
            from **{first}** to **{second}**.
            """
        )

    else:

        st.info(
            f"""
            The AI estimated a similar expression both times:
            **{second}**.
            """
        )

    st.subheader(
        "But how do YOU feel now?"
    )

    final_feeling = st.radio(
        "Choose the answer closest to how you actually feel:",
        [
            "😊 I feel better",

            "🙂 I feel a little better",

            "😐 I feel about the same",

            "😔 I still don't feel good",
        ]
    )

    if st.button(
        "Finish 💙",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.final_feeling = final_feeling

        st.session_state.stage = "goodbye"

        st.rerun()


# ============================================================
# GOODBYE
# ============================================================

elif st.session_state.stage == "goodbye":

    st.title("💙 Thank you for checking in")

    feeling = st.session_state.final_feeling

    if "better" in feeling.lower():

        st.success(
            """
            I'm glad you're feeling better.

            Take care of yourself, and remember that
            talking about things can be a good first step. 💙
            """
        )

    elif "same" in feeling.lower():

        st.info(
            """
            That's okay.

            You don't have to feel better immediately.
            Be patient with yourself. 💙
            """
        )

    else:

        st.info(
            """
            I'm sorry you're still having a difficult moment.

            If you need support, consider talking with
            someone you trust. 💙
            """
        )

    st.write(
        "Thank you for spending a moment with EmotiCare."
    )

    if st.button(
        "🔄 Start Again",
        use_container_width=True
    ):

        for key, value in DEFAULTS.items():

            st.session_state[key] = value

        st.rerun()
