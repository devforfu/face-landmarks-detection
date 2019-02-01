function init() {
    getCameraAccess();
}

let videoEnabled = false;
let drawingCanvas = null;

function getCameraAccess() {
    let video = document.querySelector("#video-element");
    if (navigator.mediaDevices.getUserMedia) {
        (navigator.mediaDevices
            .getUserMedia({video: true})
            .then((stream) => {
                video.srcObject = stream;
                videoEnabled = true
            })
            .catch((err) => {
                const message = "Cannot detect landmarks without camera access :(";
                clearMessages();
                writeMessage(message, "warning");
            })
        );
    }
}

function writeMessage(message, mode="info") {
    let node = document.createElement("li");
    let classes = ["text-large", "banner", mode];
    node.textContent = message;
    node.className = classes.join(" ");
    let messages = document.querySelector("#messages");
    messages.appendChild(node);
}

function clearMessages() {
    let messages = document.querySelector("#messages");
    for (let i = 0; i < messages.children.length; i++) {
        messages.removeChild(messages.children[i]);
    }
}

function createDrawingContext(width, height) {
    if (drawingCanvas === null) {
        let canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        let preview = document.querySelector("#image-preview");
        preview.appendChild(canvas);
        drawingCanvas = canvas;
    }
    return drawingCanvas.getContext('2d');
}

function takeSnapshot() {
    if (!videoEnabled) { return }
    let [w, h] = [640, 480];
    let context = createDrawingContext(w, h);
    let video = document.querySelector("#video-element");
    context.drawImage(video, 0, 0, w, h);
    enableDetectionButton();
}

function enableDetectionButton() {
    let button = document.querySelector("button#send-to-server");
    let classes = button.className.split(" ");
    classes.splice(classes.findIndex((x) => x === "disabled"), 1);
    button.className = classes.join(" ");
}

function detectLandmarks() {
    if (drawingCanvas !== null) {
        sendToServer(drawingCanvas.toDataURL());
    }
}

function sendToServer(data) {
    const url = `${document.location.href}detect`;
    const promise = fetch(url, {
        method: 'post',
        headers: {'Content-Type': 'application/json; charset=utf-8'},
        body: JSON.stringify({imgBase64: data})
    });
    promise.then(r => r.json()).then(data => {
        if (drawingCanvas === null) { return }
        const ok = data['success'];
        if (!ok) {
            writeMessage('Cannot detect any face on the picture');
            return
        }

        const cx = drawingCanvas.getContext('2d');
        cx.fillStyle = "green";
        cx.strokeStyle = "blue";
        cx.lineWidth = 3;

        const result = data['result'];
        for (let i = 0; i < result.length; i++) {
            const [x, y, w, h] = result[i]['box'];
            cx.strokeRect(x, y, w, h);
            const [xs, ys] = [result[i]['x'], result[i]['y']];
            for (let j = 0; j < xs.length; j++) {
                cx.beginPath();
                cx.arc(xs[j], ys[j], 2,0,2*Math.PI,true);
                cx.fill();
            }
        }
    }).catch(err => {
        console.log(`Detection error! ${err}`);
    })
}