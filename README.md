# Riskguard 
## About 
This project started off as a student innovation project for Western Cyber Societies Mainframe section. The goal was to demonstrate the ability to integrate modern machine learning fraud detection into a legacy (legendary) mainframe enviroment. The implementation was intially on a LinuxONE community cloud mainframe using s390x architecture. The main dependancy of the project is the Nvidia Triton infernece server, which we pulled in as a docker container for free with credentials from the IBM Cloud marketplace. The project was a success and endoresed by IBM Canada, but due to the unstable nature of the LinuxONE community cloud server, a lot of work was lost.

I decided that since a significant amount of the work was already done, I would rebuild it in a WSL environment. I was further encouraged to build this to take advantage of the CUDA capabilties of Triton working on my Nvidia 4060RTX. This project is a rebuild and work in progress, as well as an outlet for experimentation and continuous improvement.

