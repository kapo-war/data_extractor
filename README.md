# data_extractor
code for extract state, action


# init
First, install the library to pip package...
We changed the SC2 reading file from the original pysc2 library for supporting multiple versions of SC2, so you can download file from following link. 
//TODO: add link

# extract state info from replay data
you have to change competition_name, replays_base_path, output_base_path in scheduler function. Than, run the following command

python -m pysc2.bin.extract_state 
