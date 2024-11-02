from pydantic import BaseModel, Field
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import base64
import os
import cv2
from datetime import datetime
from PIL import Image
import platform
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_openai import ChatOpenAI
import logging

from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper
from io import BytesIO
from dotenv import load_dotenv, find_dotenv
import base64
import requests

load_dotenv(find_dotenv())



# github = GitHubAPIWrapper(github_repository="os.environ.get("GITHUB_REPO")", github_app_id=os.environ.get("GITHUB_APP_ID"), github_app_private_key=os.environ.get("GITHUB_APP_PRIVATE_KEY"))
# toolkit = GitHubToolkit.from_github_api_wrapper(github)
# tools = toolkit.get_tools()


class WebSearchTool(BaseModel):
    """
    This tool searches in the web using DuckDuckGoSearch API. 
    The docstring should clearly explain the tool's purpose and functionality.
    """


    search_query: str = Field(
        ..., description="The search query to be used in the query engine"
    )
    max_results: int = Field(..., description="The maximum number of results the engine should return. Suggested default: 3")


    def run(self):
        """Search the web for the given query."""
        logging.info(f"Searching for query: {self.search_query}")  # Log the search query
        results = DuckDuckGoSearchAPIWrapper(max_results=self.max_results)._ddgs_text(self.search_query)
        logging.info(f"Found {len(results)} results")  # Log the number of results found
        logging.info(f"Links found: {[r['href'] for r in results]}")
        return "\nPage Results: \n".join([str({"content": r["body"], "url": r["href"]}) for r in results])


class CameraFaceCaptureTool(BaseModel):
    """
    A tool for capturing a photo of the user's face using the device's camera.
    This tool uses OpenCV for camera access and face detection, and Pillow for image saving.
    It includes robust error handling and alternative capture methods.
    """

    save_path: str = Field(
        description="The directory path where the captured image will be saved. Default: ./selfie_images"
    )
    
    face_detection: bool = Field(
        description="Whether to use face detection before capturing the image. Default: False"
    )

    def run(self):
        """
        Captures a face image using the device's camera with OpenCV.
        Includes error handling and alternative capture methods.
        """
        # Check if we're on macOS
        if platform.system() == "Darwin":
            print("MacOS detected. If you encounter issues, you may need to grant camera permissions to your application.")

        # Try different camera indices
        for camera_index in range(3):
            cap = cv2.VideoCapture(1)
            if cap.isOpened():
                break
        
        if not cap.isOpened():
            return "Error: Could not open any camera. Please check your camera connections and permissions."

        # Load face detection classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        frame_count = 0
        max_attempts = 100  # Prevent infinite loop

        while frame_count < max_attempts:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                frame_count += 1
                continue  # Try to read the next frame

            if self.face_detection:
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) > 0:
                    # Face detected, capture this frame
                    break
            else:
                # No face detection, capture immediately
                break

            frame_count += 1

        # Release the camera
        cap.release()

        if frame_count >= max_attempts:
            return "Error: Could not capture a valid frame after multiple attempts. Please check your lighting and camera position."


        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"face_capture_{timestamp}.jpg"
        full_path = os.path.join(self.save_path, filename)

        os.makedirs(os.path.dirname(full_path), exist_ok=True)


        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.save(full_path, format="PNG")

        # return f"Face capture completed. Image saved to {full_path}"
        return full_path



class ImageDescriptionGeneratorTool(BaseModel):
    """
    A tool for generating descriptions of images using LangChain and OpenAI's GPT-4 Vision model.
    This tool accepts an image file path, converts the image to base64, and uses an LLM to generate a description.
    """

    image_path: str = Field(
        ...,
        description="The file path of the image to be described."
    )
    
    max_tokens: int = Field(
        ...,
        description="The maximum number of tokens in the generated description."
    )

    def run(self):
        """
        Generates a description of the image using LangChain and OpenAI's GPT-4 Vision model.
        """

        if not os.path.exists(self.image_path):
            return f"Error: Image file not found at {self.image_path}"


        with open(self.image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')


        chat = ChatOpenAI(
            model="gpt-4o-mini",
            max_tokens=self.max_tokens,
            temperature=0.7
        )


        messages = [
            SystemMessage(content="You are an AI assistant capable of analyzing and describing images in detail."),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Please describe this image in detail."},
                    {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ]
            )
        ]



        response = chat.invoke(messages)

        return response.content

    
class MyCustomTool(BaseModel):
    """
    A brief description of what the custom tool does. 
    The docstring should clearly explain the tool's purpose and functionality.
    """

    # Define the fields with descriptions using Pydantic Field
    example_field: str = Field(
        ..., description="Description of the example field, explaining its purpose and usage."
    )


    def run(self):
        """
        The implementation of the run method, where the tool's main functionality is executed.
        This method should utilize the fields defined above to perform its task.
        Doc string description is not required for this method.
        """

        # Your custom tool logic goes here
        # do_something(self.example_field)

        # Return the result of the tool's operation
        return "Result of MyCustomTool operation"
    

class ImageModificationTool(BaseModel):
    """
    A tool for modifying images using LangChain and OpenAI's DALL-E 3 model.
    This tool accepts an input image file path, a modification prompt,
    and generates a new modified image based on the prompt.
    """

    input_image_path: str = Field(
        ...,
        description="The file path of the input image to be modified."
    )
    
    modification_prompt: str = Field(
        ...,
        description="The text prompt describing how to modify the image."
    )
    
    output_image_path: str = Field(
        description="The file path where the modified image will be saved. Default: ./modified_images"
    )

    def run(self):
        """
        Modifies the input image using LangChain and OpenAI's DALL-E 3 model based on the given prompt.
        """
        # Check if the input image file exists
        if not os.path.exists(self.input_image_path):
            return f"Error: Input image file not found at {self.input_image_path}"

        # Read and encode the input image to base64
        with open(self.input_image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # with Image.open(self.input_image_path) as img:
        #     # Convert to PNG format
        #     img = img.convert("RGB")  # Ensure it's in RGB mode
        #     buffered = BytesIO()
        #     img.save(buffered, format="PNG")  # Save as PNG to a BytesIO object
        #     encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Create the ChatOpenAI instance for DALL-E 3
        chat = ChatOpenAI(model="gpt-4o-mini")

        from openai import OpenAI

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        def generate_image(prompt: str):
            response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
            )

            image_url = response.data[0].url
            return image_url

        def modify_image(k):
            response = client.images.create_variation(
            model="dall-e-2",
            image=open(self.input_image_path, "rb"),
            n=1,
            size="1024x1024"
            )

            image_url = response.data[0].url
            return image_url


        messages = [
            SystemMessage(content="You are an AI assistant capable of analyzing images and creating image editing instructions."),
            HumanMessage(
                content=[
                    {"type": "text", "text": f"Analyze this image and create a detailed DALL-E 3 prompt to modify it according to the following instructions: {self.modification_prompt}"},
                    {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ]
            )
        ]

        # Generate the DALL-E 3 prompt
        response = chat.invoke(messages)
        image_url = modify_image(response.content)

        response = requests.get(image_url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.save(self.output_image_path)
            return f"Modified image saved to {self.output_image_path}"
        else:
            return "Error: Failed to download the modified image"

# Example usage

if __name__ == "__main__":
    tool = CameraFaceCaptureTool(save_path="./selfie_images", face_detection=True)
    result_img = tool.run()
    print(result_img)
    tool = ImageDescriptionGeneratorTool(image_path=result_img)
    result = tool.run()
    print(result)
    tool = ImageModificationTool(
        input_image_path=result_img,
        modification_prompt="Make the person look 20 years older",
        output_image_path="./modified_images/modified_face.png"
    )
    result = tool.run()
    print(result)