# Air Quality Prediction

This repo contains models that predict air quality parameters based on WRF-Chem data.

The raw data should be .nc files stored in a single folder and formatted as `<string>_<string>_<string>-MM-DD_<string>`

Run extract_data.ipynb and then the Training notebook

The 3D ConvLSTM is adapted from Rohit Panda's implementation of the ConvLSTM https://github.com/sladewinter/ConvLSTM
