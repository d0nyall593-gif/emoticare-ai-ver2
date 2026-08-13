import streamlit as st
from PIL import Image
import torch

from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM


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

defaults = {
    "stage": "welcome",
    "first_emotion": None,
    "first_confidence": 0.0,
    "second_emotion": None,
    "second_confidence": 0.0,
    "actual_emotion": None,
    "messages": [],
    "final_feeling": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# EMOTION MODEL
# ============================================================

@st.cache_resource
def load_emotion_model():

    from transformers import pipeline

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

    model.eval()

    return tokenizer, model


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

def analyze_emotion(photo, emotion_model):

    image = Image.open(photo).convert("RGB")

    results = emotion_model(image)

    if not results:
        return None, 0.0

    best = results[0]

    emotion = best["label"].lower()

    confidence = float(best["score"]) * 100

    return emotion, confidence


# ============================================================
# REAL AI RESPONSE
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

The facial-expression AI estimated that the user's
expression may be "{emotion}".

IMPORTANT:
The camera cannot know the user's true feelings.
It only estimates the visible facial expression.

The user's own description of their feelings is
more important than the camera prediction.

Have a natural conversation with the user.

Rules:

- Listen carefully to what the user actually says.
- Respond to their latest message.
- Remember earlier messages.
- Ask relevant follow-up questions.
- Generate responses based on the actual conversation.
- Do not use prerecorded responses.
- Do not repeat the same response.
- Do not force the conversation to stay on the
  detected emotion.
- Do not claim that you know exactly how the user feels.
- Never diagnose mental health conditions.
- Never pretend to be a doctor or therapist.
- Be warm and respectful.
- Use simple language.
- Keep responses short.
- Usually use 1 to 3 sentences.
- If the user says they are okay, accept that.
- If the user says they are happy, do not insist
  that they are sad.
- If the user gives a short answer, gently ask
  a relevant follow-up question.
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

    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    return response


# ============================================================
# WELCOME
# ============================================================

if st.session_state.stage == "welcome":

    st.title("💙 EmotiCare AI")

    st.subheader(
        "A little space to check in with yourself."
    )

    st.write(
        """
        EmotiCare uses AI to estimate a facial expression
        and then gives you a real AI-powered conversation.
        """
    )

    st.info(
        """
        📸 Camera permission

        The camera is only used when you choose to take
        an emotion-check photo.
        """
    )

    st.warning(
        """
        ⚠️ Important

        Facial-expression AI cannot know exactly how
        someone feels.

        It only estimates the expression visible in
        the photo. Your own answer is more important.
        """
    )

    if st.button(
        "💙 Start Check-in",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.stage = "first_scan"

        st.experimental_rerun()


# ============================================================
# FIRST SCAN
# ============================================================

elif st.session_state.stage == "first_scan":

    st.title("📸 First Emotion Check")

    st.write(
        "Take a photo so the AI can estimate your facial expression."
    )

    photo = st.camera_input(
        "Take your first photo"
    )

    if photo:

        with st.spinner(
            "Loading the emotion AI..."
        ):

            try:

                emotion_model = load_emotion_model()

            except Exception as error:

                st.error(
                    "The emotion AI could not load."
                )

                st.exception(error)

                st.stop()

        with st.spinner(
            "Analyzing your expression..."
        ):

            try:

                emotion, confidence = analyze_emotion(
                    photo,
                    emotion_model
                )

                if emotion is None:

                    st.error(
                        "I couldn't detect an expression."
                    )

                else:

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
                        f"AI estimate confidence: "
                        f"{confidence:.1f}%"
                    )

                    st.write(
                        f"Does **{emotion}** match how you actually feel?"
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        if st.button(
                            "✅ Yes",
                            use_container_width=True
                        ):

                            st.session_state.actual_emotion = emotion
                            st.session_state.stage = "conversation"

                            st.experimental_rerun()

                    with col2:

                        if st.button(
                            "❌ No",
                            use_container_width=True
                        ):

                            st.session_state.stage = "correct_emotion"

                            st.experimental_rerun()

            except Exception as error:

                st.error(
                    "The emotion AI could not analyze the photo."
                )

                st.exception(error)


# ============================================================
# CORRECT EMOTION
# ============================================================

elif st.session_state.stage == "correct_emotion":

    st.title("💭 You know yourself best")

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

        st.experimental_rerun()


# ============================================================
# REAL AI CONVERSATION
# ============================================================

elif st.session_state.stage == "conversation":

    st.title("💬 Talk to EmotiCare")

    with st.spinner(
        "Starting the conversation AI..."
    ):

        try:

            tokenizer, chat_model = load_chat_model()

        except Exception as error:

            st.error(
                "The conversation AI could not load."
            )

            st.exception(error)

            st.stop()

    # --------------------------------------------------------
    # OPENING
    # --------------------------------------------------------

    if len(st.session_state.messages) == 0:

        opening = (
            "I'd like to hear what's on your mind. "
            f"The camera estimated a possible "
            f"{st.session_state.actual_emotion} expression, "
            "but you know yourself better. "
            "Why do you think you're feeling this way?"
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": opening
            }
        )

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

    # --------------------------------------------------------
    # USER MESSAGE
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

                st.error(
                    "The AI had trouble generating a response."
                )

                st.exception(error)

                response = (
                    "I'm having a little trouble thinking "
                    "right now. Could you tell me a little more?"
                )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        st.experimental_rerun()

    # --------------------------------------------------------
    # FINISH
    # --------------------------------------------------------

    st.divider()

    st.write(
        "When you're ready to finish:"
    )

    if st.button(
        "💙 How do you feel now?",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.stage = "second_scan"

        st.experimental_rerun()


# ============================================================
# SECOND SCAN
# ============================================================

elif st.session_state.stage == "second_scan":

    st.title("💙 How do you feel now?")

    st.write(
        """
        Take another photo so EmotiCare can make
        a second facial-expression estimate.
        """
    )

    st.caption(
        "Remember: this is only an AI estimate."
    )

    photo = st.camera_input(
        "Take your final photo"
    )

    if photo:

        with st.spinner(
            "Analyzing your expression again..."
        ):

            try:

                emotion_model = load_emotion_model()

                emotion, confidence = analyze_emotion(
                    photo,
                    emotion_model
                )

                if emotion is None:

                    st.error(
                        "I couldn't analyze the second photo."
                    )

                else:

                    st.session_state.second_emotion = emotion
                    st.session_state.second_confidence = confidence
                    st.session_state.stage = "result"

                    st.experimental_rerun()

            except Exception as error:

                st.error(
                    "The second emotion scan failed."
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
            f"AI estimate: "
            f"{st.session_state.first_confidence:.1f}%"
        )

    with col2:

        st.subheader("Now")

        st.write(
            f"{EMOTION_EMOJIS.get(second, '🙂')} "
            f"**{second.capitalize()}**"
        )

        st.caption(
            f"AI estimate: "
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
            The AI estimated a similar facial expression
            both times: **{second}**.
            """
        )

    st.subheader(
        "But how do YOU feel now?"
    )

    final_feeling = st.radio(
        "Choose the answer that feels closest:",
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

        st.experimental_rerun()


# ============================================================
# GOODBYE
# ============================================================

elif st.session_state.stage == "goodbye":

    st.title("💙 Thank you for checking in")

    feeling = st.session_state.final_feeling

    if "better" in feeling.lower():

        st.success(
            """
            I'm glad you're feeling better. 💙

            Remember to be kind to yourself.
            """
        )

    elif "same" in feeling.lower():

        st.info(
            """
            That's okay. 💙

            You don't have to feel better immediately.
            Give yourself some time.
            """
        )

    else:

        st.info(
            """
            I'm sorry you're still having a difficult moment.

            Consider talking to someone you trust if you
            need some extra support. 💙
            """
        )

    st.write(
        "Thank you for spending a moment with EmotiCare."
    )

    if st.button(
        "🔄 Start Again",
        use_container_width=True
    ):

        for key in list(defaults.keys()):

            if key in st.session_state:
                del st.session_state[key]

        st.experimental_rerun()
