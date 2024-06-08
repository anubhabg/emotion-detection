document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureButton = document.getElementById('capture');
    const recaptureButton = document.getElementById('recapture');
    const capturedImageInput = document.getElementById('captured_image');
    const capturedImage = document.getElementById('captured-image');
    const videoContainer = document.getElementById('video-container');
    const recaptureOptions = document.getElementById('recapture-options');

    // Access the webcam
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
            video.srcObject = stream;
            video.play();
        }).catch(err => {
            console.error("Error accessing the webcam: ", err);
        });
    } else {
        console.error("getUserMedia not supported by this browser.");
    }

    // Capture the image from the webcam
    captureButton.addEventListener('click', () => {
        const context = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageDataURL = canvas.toDataURL('image/png');
        capturedImageInput.value = imageDataURL; // Set the captured image data to the hidden input

        // Display the captured image
        capturedImage.src = imageDataURL;
        capturedImage.style.display = 'block';

        // Hide the video and capture button
        videoContainer.style.display = 'none';

        // Show the recapture options
        recaptureOptions.style.display = 'block';
    });

    // Recapture the image
    recaptureButton.addEventListener('click', () => {
        // Show the video and capture button
        videoContainer.style.display = 'block';

        // Hide the captured image and recapture options
        capturedImage.style.display = 'none';
        recaptureOptions.style.display = 'none';

        // Clear the hidden input
        capturedImageInput.value = '';
    });
});