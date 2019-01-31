function init() {
    getCameraAccess();
}

function getCameraAccess() {
    let video = document.querySelector("#video-element");
    if (navigator.mediaDevices.getUserMedia) {
        (navigator.mediaDevices
            .getUserMedia({video: true})
            .then((stream) => {
                video.srcObject = stream;
            })
            .catch((error) => {
                console.log(`Cannot access camera: ${error}`);
            })
        );
    }
}