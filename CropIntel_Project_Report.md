# CropIntel Project Report

## 1. Introduction

### 1.1 Objective of the project
The objective of the **CropIntel** project is to design and implement an AI‑based web application that can automatically detect plant diseases from leaf images. The system aims to assist farmers, agronomists, and students in early diagnosis of plant diseases using image analysis, to provide a simple browser-based interface where users can upload a photo of an affected leaf and obtain instant predictions, to leverage state-of-the-art deep learning models for plant disease classification while keeping the deployment lightweight and accessible on commodity hardware, and to serve as a foundation for future research and deployment, including integration with agronomic decision support systems and mobile-based advisory tools. This work builds upon earlier research showing that deep learning can effectively detect plant diseases from image data [8].

### 1.2 Brief description of the project
CropIntel is a **two-tier web application** consisting of a frontend built with React and Tailwind CSS that provides a modern, responsive user interface through which users can upload a plant leaf image via drag-and-drop or file selection, view a preview, and trigger AI analysis with one click, and a backend implemented using Flask that exposes REST APIs, loads a pre-trained **MobileNetV2-based plant disease identification model** (from Hugging Face Transformers and PyTorch) once at startup, processes incoming images, runs inference, and returns structured JSON predictions with confidence scores.

The typical workflow is that the user opens the web app, uploads an image, the frontend sends the image as `multipart/form-data` to `/predict`, the backend preprocesses the image and runs the model, the predictions are sorted by confidence, and the frontend displays the top disease class along with a ranked list of likely diseases and corresponding confidence bars.

### 1.3 Technology used

#### 1.3.1 H / W Requirement
CropIntel can be developed and deployed on a modern development or server machine with a multi-core processor (for example, an Intel i5/i7 or AMD equivalent), at least 8 GB of RAM with 16 GB recommended for smoother model loading and development, and approximately 5–10 GB of free storage space for Python and Node environments, dependencies, and model weights. A GPU is optional because the current implementation runs on CPU, but a CUDA-enabled GPU can be used to speed up inference when available. On the client side, the application only requires a desktop, laptop, or smartphone with a modern web browser such as Chrome, Edge, Firefox, or Safari and a stable internet or local network connection to reach the Flask backend.

#### 1.3.2 Software Requirement
For development, CropIntel can be run on Windows, Linux, or macOS. The backend stack consists of Python 3.x with Flask as the REST API framework [1], `flask-cors` to allow cross-origin requests from the React frontend, the `transformers` library from Hugging Face to load the MobileNetV2 plant disease model [3, 4], the `torch` and `torchvision` libraries from PyTorch for deep learning [2], and `pillow` for image loading and basic preprocessing, with all dependencies managed via `requirements.txt`. The frontend stack uses Node.js and npm with React 18 for building the user interface [5], Vite as the development server and build tool [7], Tailwind CSS as a utility-first styling framework [6], axios as the HTTP client for API requests, and lucide-react for icons. Additional tools include browser developer tools for debugging and command-line utilities such as the terminal, npm scripts, and Python virtual environments.

### 1.4 Organization Profile (if applicable)
This project was carried out in the context of the **[Department Name]**, **[Institute/University Name]**, which focuses on combining **computer science** and **agricultural engineering** to develop practical solutions for farmers. The institution encourages student projects that apply modern AI/ML techniques to real-world societal problems such as crop productivity, disease management, and sustainable agriculture.

You can replace the placeholders **[Department Name]** and **[Institute/University Name]** with your actual details.

---

## 2. Design Description

### 2.1 Flow Chart (conceptual description)

You can draw the flow chart based on the following sequence of actions. The flow starts when the user opens the CropIntel web app in a browser and the React user interface is loaded. The user then uploads a plant leaf image, either via drag-and-drop or by using the file chooser. The frontend validates that an image file has been provided and that it is of an acceptable type; if no file is present or the type is invalid, an error message is shown and the user is prompted to try again, whereas a valid file allows the process to continue. Once validation succeeds, the React application sends a `POST /predict` request to the backend, including the image as `multipart/form-data` using axios. The Flask backend receives this request, checks that the plant disease model has been loaded successfully and that the image file is present, and then performs image preprocessing by reading the uploaded bytes into a PIL image, converting it to RGB if necessary, resizing it to 224×224 pixels, normalizing the pixel values, and converting the result into a PyTorch tensor. The preprocessed tensor is then passed through the MobileNetV2 plant disease classification model, and a softmax operation is applied to obtain probabilities for all disease classes. After inference, the backend prepares a response by creating a list of labels and scores, sorting this list in descending order of score, identifying the top prediction, and returning a JSON object containing a success flag, the complete list of predictions, and the top prediction. The frontend receives this response and displays the detected disease label, a confidence bar, and the top-N predictions, or an appropriate error message if any problem occurred. The flow ends when the user reviews the results, although they can reset the form and upload another image, which causes the same sequence of steps to repeat.

### 2.2 Data Flow Diagrams (DFDs)

You can draw the data flow diagrams using the descriptions below. At Level 0, the context diagram shows a single external entity, the user (who may be a farmer, agronomist, or student), interacting with the CropIntel system. The user sends a plant leaf image to the system, and CropIntel returns disease predictions together with the associated confidence scores.

At Level 1, the DFD decomposes the system into three main processes. Process P1, Image Upload and Validation, takes the image file from the user as input, performs basic validation, and produces either validated image data or an error message, with the image optionally held in temporary in-memory storage. Process P2, the Disease Prediction Engine, receives the validated image, performs preprocessing and model inference using PyTorch and the Transformers library, and relies on an in-memory model store that contains the preloaded MobileNetV2 plant disease model weights and configuration. Process P3, Results Formatting and Presentation, consumes the raw model probabilities and labels, transforms them into a structured JSON response with the top prediction and the full prediction list, and hands this information to the frontend, which renders the results as charts, bars, and text for the user.

At runtime, the main data stores are **in-memory model parameters** and short-lived **request data**; there is no persistent external database in the current implementation.

### 2.3 Entity Relationship Diagram (E-R Diagram)

Although the current system does **not** use a database, a logical ER model is useful for future extensions such as logging predictions and managing user accounts. A conceptual ER design could define a User entity (optional for future work) with attributes such as `UserID`, `Name`, `Email`, and `Role` (farmer, expert, or admin). A PlantImage entity would hold information about each uploaded image, including `ImageID`, `FileName`, `UploadTime`, `Resolution`, `Format`, and `UploadedBy` as a foreign key to the corresponding user record. A Prediction entity would represent each model output, with attributes such as `PredictionID`, a foreign key `ImageID` linking back to the corresponding PlantImage, the `DiseaseLabel`, the `ConfidenceScore`, the `Rank` of the prediction, and the `PredictedAt` timestamp. A DiseaseClass entity would serve as a reference table containing `DiseaseLabel` as the primary key along with attributes such as `Crop`, `Description`, and `TypicalSymptoms`.

In this conceptual design, one User can upload many PlantImage records, each PlantImage can have multiple Prediction records, and each Prediction is associated with one DiseaseClass. In the current implementation, these “entities” exist only as in-memory objects, such as Python dictionaries and lists in the Flask response, rather than as persisted database tables.

---

## 3. Project Description

### 3.1 Data Base

The present version of CropIntel is intentionally lightweight and **does not use a persistent database**. All processing is done in memory on the backend, where image tensors and model outputs are held temporarily, and as transient state on the frontend, where React `useState` hooks store the selected image, the preview URL, the prediction results, and any error messages.

Although no physical database is currently used, the design is **database-ready**. A database can be introduced later to store historical predictions for analytics, to manage registered users and access control, and to maintain a curated catalog of diseases along with their recommended treatments. For future work, a relational database such as **PostgreSQL** or **SQLite** for single-node deployments would be suitable choices.

### 3.2 Table Description (for a future DB design)

If and when a database is added, the logical design can be implemented using a small set of relational tables. A `users` table would store user-specific information such as a primary key `user_id` of type integer, the `name` and `email` fields with the email constrained to be unique, a `role` field indicating whether the user is a farmer, expert, or admin, and a `created_at` timestamp. A `plant_images` table would record each uploaded image, using an integer primary key `image_id`, an optional foreign key `user_id` referencing `users.user_id`, the `file_path` where the image is stored, an `uploaded_at` datetime field, the `original_filename` as provided by the user, and additional metadata such as `width`, `height`, and `format`.

Predictions generated by the model would be represented in a `predictions` table, which would have an integer primary key `prediction_id`, a foreign key `image_id` pointing to `plant_images.image_id`, a `disease_label` field that references the disease catalog, a numeric `confidence_score`, a numeric `rank` indicating the ordering among multiple predictions for the same image, and a `predicted_at` timestamp. The disease catalog itself would be captured in a `disease_classes` table, where `disease_label` serves as the primary key and additional fields such as `crop`, `description`, and `typical_symptoms` describe each disease class in detail. Collectively, these tables mirror the data structures that are already used logically in the API responses.

### 3.3 File / Database Design

Since there is no physical database yet, the design focuses on the **file structure** of the application. On the backend, contained in the `backend/` directory, the central file is `app.py`, which initializes the Flask application and configures CORS, loads the pre-trained MobileNetV2 plant disease model and image processor, implements the `/health` endpoint for health checks and the `/predict` endpoint to handle image uploads, preprocessing, inference, and JSON responses, and uses logging to support debugging and monitoring. The `requirements.txt` file complements this by listing all backend Python dependencies with pinned versions to ensure consistent environments.

On the frontend side, contained in the `frontend/` directory, the key file is `src/App.jsx`, which defines the main React component. This component manages the state variables `selectedImage`, `imagePreview`, `loading`, `result`, and `error`, implements image upload handling via both file input and drag-and-drop, uses axios to call the `http://localhost:5000/predict` endpoint on the backend, and displays the analysis results and confidence bars. The `package.json` file manages Node and React dependencies and scripts such as `dev`, `build`, and `preview`, while configuration files like `tailwind.config.js`, `postcss.config.js`, and `vite.config.js` define the build process and styling behavior. This modular structure separates concerns between the user interface and the inference logic and makes the system maintainable and extendable.

---

## 4. Input/Output Form Design

The input form in the React user interface is organized around an upload section that appears as a card titled “Upload Plant Image”. Within this section, users see a drag-and-drop area with a dashed border where they can drop image files, as well as a hidden file input that is bound to a “click to browse” label for users who prefer to select files via the file chooser. The form accepts only image formats matching `image/*`, such as JPG, PNG, and WEBP. Once an image is selected, the interface immediately displays a preview of the uploaded image, and the user can then interact with two main buttons: one to analyze the disease, which triggers the AI prediction, and another to reset the form, which clears the current selection and any displayed results.

The output or results section is presented as a separate card titled “Analysis Results”. This section reflects several possible states: in the idle state, the user sees a message prompting them to upload an image; during processing, a loading state shows a spinner and descriptive text such as “Analyzing your plant…”; and if an error occurs, a clearly styled alert displays a descriptive message indicating issues such as a missing image or an unavailable backend. When analysis succeeds, the section displays a top prediction block that shows the “Identified Disease” using the label from `result.top_prediction.label` together with a confidence progress bar derived from `result.top_prediction.score`, as well as an additional block listing the top few predictions with their confidence scores and corresponding smaller progress bars. Overall, the form emphasizes clarity and usability, requiring minimal user input beyond supplying an image and presenting the output in a visually intuitive way.

---

## 5. Testing & Tools used (if applicable)

### Testing Approach

Since no automated test suite is present in the repository, testing was primarily **manual and exploratory**. On the backend side, the `/health` endpoint was verified to return a status of `healthy` with `model_loaded = True`, and the `/predict` endpoint was exercised using tools such as Postman and cURL with a variety of image files at different resolutions and formats. Additional backend checks included sending requests without the `image` field to confirm that an HTTP 400 error is returned, testing with empty filenames, and using very large images to observe performance and error handling, while monitoring the backend logs to ensure that the model loads correctly and that any prediction errors are captured.

For the frontend, functional testing in the browser focused on verifying file selection and drag-and-drop behavior, ensuring that the preview image is displayed correctly once a file is chosen, and confirming that successful backend responses lead to the correct rendering of the top prediction and the full list of predictions. Testing also covered failure scenarios, such as network issues or cases where the backend model is not loaded, to ensure that clear error messages are shown to the user. Cross-browser checks were carried out on recent versions of Chrome, Edge, and Firefox to confirm that the interface behaves consistently.

### Tools used for Testing

The main tools used for testing were the browser developer tools (such as Chrome and Edge DevTools), which allowed inspection of network requests, console logs, and component behavior, along with Postman and cURL for direct REST API testing of the Flask endpoints. The Flask debug server logs were also used extensively to track incoming requests, errors, and model loading events. Future work may include adding **unit tests**, for example using pytest on the backend, and **component tests**, for example with React Testing Library on the frontend, to automate regression testing and improve overall reliability.

---

## 6. Implementation & Maintenance (if applicable)

### Implementation

From an implementation perspective, the backend is realized as a single Flask application defined in `app.py` [1]. At startup, the application loads the Hugging Face `AutoImageProcessor` and `AutoModelForImageClassification` using the model `"linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"` [3, 4], and keeps this model in memory for subsequent requests. Image preprocessing is carried out using **Pillow** and **PyTorch** [2], where uploaded images are converted from PIL format into normalized tensors suitable for the model. Inference is executed within a `torch.no_grad()` block to avoid gradient computations and thereby reduce overhead and improve performance.

The frontend implementation is based on React functional components and the `useState` hook [5]. It uses Tailwind CSS classes to control layout and styling [6], which helps achieve a responsive design across different screen sizes. Axios is used to send `POST /predict` requests to the backend with the uploaded image encoded as `multipart/form-data`, and the frontend logic clearly distinguishes between loading, error, and success states, updating the user interface accordingly.

### Maintenance

Ongoing maintenance of CropIntel focuses on managing dependencies, updating the model, and extending functionality in a controlled way. Dependencies are pinned in `requirements.txt` and `package.json` to ensure reproducible builds across different environments, but regular updates to libraries such as Flask, transformers, torch, and React will still be necessary to address security issues and benefit from performance improvements.

The model itself is loaded via its name from Hugging Face, which means that swapping to a newer or custom fine-tuned model can be achieved by updating the model identifier and verifying that the label mappings remain correct. The codebase is intentionally small and modular, which makes it straightforward to add new endpoints, such as those for batch predictions, to integrate a database or user authentication mechanism, or to enhance the user interface with additional information like disease descriptions and recommended actions. For deployment and monitoring in a production environment, the application can be containerized with Docker and the existing logging setup, based on Python’s `logging` module, can be connected to centralized monitoring tools.

---

## 7. Conclusion and Future Work

### Conclusion
CropIntel demonstrates a practical application of **deep learning for plant disease detection** using an accessible web interface. By combining a pre-trained MobileNetV2 model with a lightweight Flask backend and a React-based frontend, the project delivers **near real-time** predictions from simple leaf images. The current prototype validates that such AI tools can aid early diagnosis and potentially reduce crop losses.

### Future Work
Planned and possible enhancements to CropIntel include integrating a **persistent database** to store prediction histories and user profiles, extending the frontend so that it presents detailed disease descriptions and recommended control measures alongside the predictions, and packaging the solution as a **mobile-friendly progressive web app (PWA)** or native application for use directly in the field. In addition, the model can be fine-tuned on **local or custom datasets** to improve accuracy for particular regions and crops, and the system can be deployed on cloud infrastructure to support larger-scale use, accompanied by field trials with farmers and experts to validate its effectiveness.

---

## 8. Outcome

The current outcome of the project is a **fully working prototype** of an AI-powered plant disease detection web application that has been implemented and tested locally. The system successfully integrates a pre-trained deep learning model with a modern React frontend and provides practical, usable predictions for uploaded plant leaf images. In its present form, the project is ready to be extended into a more formal research study and/or a deployed field trial, depending on the extent of additional data collection, validation, and integration work that is carried out.

Depending on the specific requirements of the academic or institutional context, this outcome can be characterized as a research-oriented prototype, a deployment-ready demonstration, or the basis for a formal research paper, and the wording in this section can be adapted accordingly.

---

## 9. Bibliography

[1] Flask Documentation, *Flask Web Development Framework*. Available at: https://flask.palletsprojects.com/  
[2] PyTorch Documentation, *An Open Source Machine Learning Framework*. Available at: https://pytorch.org/  
[3] Hugging Face Transformers, *State-of-the-art Machine Learning for Pytorch and TensorFlow 2.0*. Available at: https://huggingface.co/docs/transformers  
[4] Linkan Jarad, *MobileNetV2 Plant Disease Identification Model* (Hugging Face Model Hub), Available at: https://huggingface.co/ (model: `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`).  
[5] React Documentation, *A JavaScript Library for Building User Interfaces*. Available at: https://react.dev/  
[6] Tailwind CSS, *Utility-First CSS Framework for Rapid UI Development*. Available at: https://tailwindcss.com/  
[7] Vite, *Next Generation Frontend Tooling*. Available at: https://vitejs.dev/  
[8] S. P. Mohanty, D. P. Hughes, and M. Salathé, “Using Deep Learning for Image-Based Plant Disease Detection,” *Frontiers in Plant Science*, vol. 7, 2016.
