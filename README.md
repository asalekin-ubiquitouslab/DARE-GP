# DARE-GP
Develop universal adversarial perturbations that protect user emotion in speech without disrupting transcription. 

This code uses the RAVDESS (https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) and TESS (https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) datasets. Use the fix_ravdess.py and fix_tess.py scripts to rename these to the correct naming convention.

This code requires very specific library versions (specifically for the vosk/kaldi transcription code). See the docker/Dockerfile file for specifics. 

To cite this work, please include this reference:

@article{testa2023privacy,
  title={Privacy against Real-Time Speech Emotion Detection via Acoustic Adversarial Evasion of Machine Learning},
  author={Testa, Brian and Xiao, Yi and Sharma, Harshit and Gump, Avery and Salekin, Asif},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={7},
  number={3},
  pages={1--30},
  year={2023},
  publisher={ACM New York, NY, USA}
}
