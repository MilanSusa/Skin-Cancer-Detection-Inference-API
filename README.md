# Skin Cancer Detection Inference API

Skin cancer detection inference API provides user the option of performing inference on a given dermatoscopic image in 
order to determine which important diagnostic categories in the realm of pigmented lesions are present in the picture. 
Important diagnostic categories in the realm of pigmented lesions are:
* actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)
* basal cell carcinoma (bcc)
* benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl)
* dermatofibroma (df)
* melanoma (mel)
* melanocytic nevi (nv)
* vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc)

## Prerequisites

Download [model][1] and put it in the `pretrained_models` directory.

[1]: https://drive.google.com/open?id=1UBux4baGx74mSNwxRz7YtvWD5sKEEMMd

## Running locally

Navigate to the root directory and execute the following line to start a web server for the application:

    flask run 
    
In case you want to change the default port, set the `FLASK_RUN_PORT` environment variable to meet your needs.
    
Application server will start on default port so you can try out the API by running:

    curl -X POST 'http://localhost:{port}/api/v1/inference' \
    -H 'Content-Type: multipart/form-data' \
    -F 'image=@{path_to_your_image}'

while replacing `{port}` with default port and `{path_to_your_image}` with the path of the image you want to perform 
inference on.

## License

Licensed under the [MIT License](LICENSE).
