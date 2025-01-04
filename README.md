Since the covid_resnet.keras model in the ai directory is a large file, you need to use lfs to fetch it by using the command
"git lfs pull"

Make sure that you have downloaded Git LFS and use the command "git lfs install" in your repository first

or you can download the model from this link:
https://binusianorg-my.sharepoint.com/personal/nathaniel_alexander_binus_ac_id/_layouts/15/guestaccess.aspx?share=ETSjBcUkM8pDmEwdutM2taIBf1yK4f6YeijLnUIoHg8h_w&e=IT8Zcx

How to run:
1. Activate the Python Virtual Environment for Flask: -> "path_to_venv\venv\Scripts\activate.bat"
2. Navigate to the Flask server directory -> "cd CovidX-main/ai"
3. Run the server -> "python Flask_Covid.py"
4. Open another terminal for Next.js (frontend)
5. Navigate to the Next.js app directory -> "cd CovidX-main/covid-x"
6. Run Next.js app -> "npm run dev"
7. open the app in your browser -> http://localhost:3000

Here are the colab links for training the model:
Machine Learning COVID-19: https://colab.research.google.com/drive/1nMRoXiTRNg71avgMGv2CT-k0hRembxh3?usp=sharing
ResNet Image Classification COVID-19: https://colab.research.google.com/drive/1qD7ObckOf8iBRvvlbxB76444xE9mEb1m?usp=sharing
VGG Image Classification COVID-19: https://colab.research.google.com/drive/14-7nR_NVxLX90uOiw680Qe9Ui5kRTKsn?usp=sharing
