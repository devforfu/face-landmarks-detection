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
            .catch((error) => {
                console.log(`Cannot access camera: ${error}`);
            })
        );
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
        console.log(data);
    }).catch(err => {
        console.log(`Detection error! ${err}`);
    })
}