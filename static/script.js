document.addEventListener("DOMContentLoaded", () => {
    const questionInput = document.getElementById("question-input");
    const sendButton = document.getElementById("send-button");
    const chatHistory = document.getElementById("chat-history");
    const thinkingTemplate = document.getElementById("thinking-template");
    const fileUpload = document.getElementById("file-upload");
    const uploadButton = document.getElementById("upload-button");
    
    let uploadedDocuments = [];

    // Auto-resize textarea logic
    const autoResize = () => {
        questionInput.style.height = "auto";
        questionInput.style.height = Math.min(questionInput.scrollHeight, 200) + "px";
    };

    questionInput.addEventListener("input", autoResize);

    // Scroll to the absolute bottom of the chat container
    const scrollToBottom = () => {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    };

    // Helper to create a new message bubble
    const createMessageBubble = (role, contentHTML) => {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${role}`;
        
        const contentDiv = document.createElement("div");
        contentDiv.className = "message-content";
        contentDiv.innerHTML = contentHTML;
        
        msgDiv.appendChild(contentDiv);
        return msgDiv;
    };

    // Helper to escape HTML to prevent XSS in user input
    const escapeHTML = (str) => {
        const p = document.createElement("p");
        p.appendChild(document.createTextNode(str));
        return p.innerHTML;
    };

    const submitQuestion = async () => {
        const question = questionInput.value.trim();
        if (!question) return;

        // 1. Immediately display user's message
        const escapedQuestion = escapeHTML(question);
        const userBubble = createMessageBubble("user", escapedQuestion);
        chatHistory.appendChild(userBubble);

        // Clear input and reset height immediately
        questionInput.value = "";
        autoResize();

        // 2. Display "Thinking..." bubble
        const thinkingHTML = thinkingTemplate.innerHTML;
        const assistantBubble = createMessageBubble("assistant", thinkingHTML);
        chatHistory.appendChild(assistantBubble);
        scrollToBottom();

        // Disable input
        questionInput.disabled = true;
        sendButton.disabled = true;

        try {
            // 3. Call backend POST /ask
            const response = await fetch("/ask", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ question: question })
            });

            if (!response.ok) {
                throw new Error("Failed to communicate with the server.");
            }

            const data = await response.json();
            const contentDiv = assistantBubble.querySelector(".message-content");

            // 4. Replace Thinking bubble with actual response
            // Ignoring backend metrics entirely as per requirements
            if (data.success) {
                contentDiv.textContent = data.answer;
            } else {
                assistantBubble.classList.add("error");
                contentDiv.textContent = data.answer || "An unexpected error occurred.";
            }

        } catch (error) {
            console.error("Fetch error:", error);
            const contentDiv = assistantBubble.querySelector(".message-content");
            assistantBubble.classList.add("error");
            contentDiv.textContent = "Network error or server is unreachable. Please try again.";
        } finally {
            // Restore UI interactivity and ensure scroll is updated
            questionInput.disabled = false;
            sendButton.disabled = false;
            scrollToBottom();
            questionInput.focus();
        }
    };

    // Event Listeners
    sendButton.addEventListener("click", submitQuestion);

    uploadButton.addEventListener("click", () => {
        fileUpload.click();
    });

    fileUpload.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Reset input so the same file can be uploaded again
        e.target.value = '';

        const escapedName = escapeHTML(file.name);
        const thinkingHTML = thinkingTemplate.innerHTML;
        const uploadBubble = createMessageBubble("assistant", `Uploading and processing <b>${escapedName}</b>...<br><br>${thinkingHTML}`);
        chatHistory.appendChild(uploadBubble);
        scrollToBottom();

        questionInput.disabled = true;
        sendButton.disabled = true;
        uploadButton.disabled = true;

        try {
            const formData = new FormData();
            formData.append("file", file);

            const response = await fetch("/upload", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error("Failed to communicate with the server.");
            }

            const data = await response.json();
            const contentDiv = uploadBubble.querySelector(".message-content");

            if (data.success) {
                uploadedDocuments.push({
                    id: data.document_id,
                    name: data.document_name
                });
                
                contentDiv.innerHTML = `Successfully processed <b>${escapeHTML(data.document_name)}</b>.<br>Generated ${data.chunk_count} chunks and ${data.embedding_count} vectors.<br>You can now ask questions about it!`;
                uploadBubble.classList.remove("error");
            } else {
                uploadBubble.classList.add("error");
                contentDiv.textContent = data.message || "Failed to process the document.";
            }
        } catch (error) {
            console.error("Upload error:", error);
            const contentDiv = uploadBubble.querySelector(".message-content");
            uploadBubble.classList.add("error");
            contentDiv.textContent = "Network error or server is unreachable during upload. Please try again.";
        } finally {
            questionInput.disabled = false;
            sendButton.disabled = false;
            uploadButton.disabled = false;
            scrollToBottom();
            questionInput.focus();
        }
    });

    questionInput.addEventListener("keydown", (e) => {
        // If Enter is pressed WITHOUT shift, send message
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault(); // Prevent default newline
            submitQuestion();
        }
        // If Shift+Enter, allow default behavior (newline), input event will trigger resize naturally
    });
    
    // Initial focus on load
    questionInput.focus();
});
