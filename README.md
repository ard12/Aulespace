# AULE Computer Vision Challenge

My solution to a four-part computer vision take-home, plus the follow-up set after the interview.

1. **Rotation:** how far has the image been turned?
2. **Crop localisation:** where in the drawing did this window come from, and how should the camera move until the marker is in view?
3. **Side view:** what does a camera 100 cm away and 22.5 degrees to the right see?
4. **Return:** move that camera back to the front view while staying 100 cm from the centre.

**Follow-up:** tasks 2 and 4 redone so the camera never trusts its own movement. After every move it works out where it is from the image alone.

**[Open the notebook in Google Colab](https://colab.research.google.com/github/ard12/Aulespace/blob/main/AULE_ComputerVisionChallenge/AULE_ComputerVisionChallenge_Solution.ipynb)**

The notebook carries its own copy of the code and reference image. Runtime > Run all takes about a minute, and the saved outputs can be read without running it.

| | |
|---|---|
| Notebook | [`AULE_ComputerVisionChallenge_Solution.ipynb`](AULE_ComputerVisionChallenge/AULE_ComputerVisionChallenge_Solution.ipynb) |
| Method, results and limitations | [`AULE_ComputerVisionChallenge/README.md`](AULE_ComputerVisionChallenge/README.md) |
| Code | [`src/aulecv/`](AULE_ComputerVisionChallenge/src/aulecv) |
| Tests (242) | [`tests/`](AULE_ComputerVisionChallenge/tests) |
| Step-by-step notebooks | [`notebooks/`](AULE_ComputerVisionChallenge/notebooks) |

Everything is tested in simulation built from the supplied drawing, not on a real camera.
