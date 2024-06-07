
const send_buttom = document.querySelector('.send_bottom');

send_buttom.addEventListener('click', () => {
    const chat_area = document.querySelector('#chat_text_area');
    let chat_content = chat_area.value;
    if(chat_content !=''){
        printQuestion(chat_content);
        fetch('http://127.0.0.1:5000/predict_query_checker', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ input_sentence: chat_content })
        })
        .then(response => response.json())
        .then(data => {
            // Handle the response
            // console.log(data)
            // console.log(data.predicted_label);
            if(data.predicted_label === 'non-medical query'){
                htmldata = `<div class="display_chat_answer">
                    <p class="answer_chat">
                    I am not trained to perform any other conversation. 
                    I can only help you with skin-related problems. 
                    Please describe your symptoms, 
                    and I will assist you in identifying the cause.
                    </br>
                    If you have any skin-related queries, feel free to ask.
                    </p>
                </div>`

                const chat_display_area = document.querySelector('.chat_display_area');
                chat_display_area.insertAdjacentHTML("beforeend", htmldata);
            }
            else{
                categoryClassify(chat_content);
            }
            // printAnswer(data.predicted_label);
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }else{
        console.log('please write something');
    }
})


const categoryClassify = (chat_content) => {
    fetch('http://127.0.0.1:5001/predict_disease_category', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ input_sentence: chat_content })
    })
    .then(response => response.json())
    .then(data => {
        // Handle the response
        // console.log(data)
        // console.log(data.predicted_category);
        if(data.predicted_category === 'fever related'){
            htmldata = `<div class="display_chat_answer">
                <p class="answer_chat">
                Your Symptom information is about Fever Related. 
                But I can only help you with skin-related problems. 
                Please describe your skin-related symptoms, 
                and I will assist you in identifying the cause.
                </br>
                If you have any skin-related queries, feel free to ask.
                </p>
            </div>`
            const chat_display_area = document.querySelector('.chat_display_area');
            chat_display_area.insertAdjacentHTML("beforeend", htmldata);
        }else if(data.predicted_category === 'gastric related'){
            htmldata = `<div class="display_chat_answer">
                <p class="answer_chat">
                Your Symptom information is about Fever Related. 
                But I can only help you with skin-related problems. 
                Please describe your skin-related symptoms, 
                and I will assist you in identifying the cause.
                </br>
                If you have any skin-related queries, feel free to ask.
                </p>
            </div>`
            const chat_display_area = document.querySelector('.chat_display_area');
            chat_display_area.insertAdjacentHTML("beforeend", htmldata);
        }else if(data.predicted_category === 'others query'){
            htmldata = `<div class="display_chat_answer">
                <p class="answer_chat">
                I can only help you with skin-related problems. 
                Please describe your skin-related symptoms, 
                and I will assist you in identifying the cause.
                </br>
                If you have any skin-related queries, feel free to ask.
                </p>
            </div>`
            const chat_display_area = document.querySelector('.chat_display_area');
            chat_display_area.insertAdjacentHTML("beforeend", htmldata);
        }else{
            PredictDisease(chat_content)
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

const PredictDisease = (chat_content) => {
    fetch('http://127.0.0.1:5002/predict_skin_disease', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ input_sentence: chat_content })
    })
    .then(response => response.json())
    .then(data => {
        // Handle the response
        // console.log(data)
        // console.log(data.predicted_disease);
        printAnswer(data.predicted_disease);
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

const printQuestion = (question) => {

    htmldata = `<div class="display_chat_question">
                    <p class="question_chat">${question}</p>
                </div>`

    const chat_display_area = document.querySelector('.chat_display_area');
    chat_display_area.insertAdjacentHTML("beforeend", htmldata);
}

const printAnswer = (answer) => {
    htmldata = `<div class="display_chat_answer">
                    <p class="answer_chat">
                    You may be affected by this skin disease ${answer}. 
                    Please consult a doctor with this report 
                    for a professional diagnosis and appropriate treatment.
                    if you have any farther
                    </br>
                    If you have any other skin-related queries, feel free to ask. 
                    </p>
                </div>`

    const chat_display_area = document.querySelector('.chat_display_area');
    chat_display_area.insertAdjacentHTML("beforeend", htmldata);
}