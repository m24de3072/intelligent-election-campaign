# Steps to process the dataset, train the model(s) and then view the results in a streamlit app.

1. Run the python script from the dataset directory `create_ac_pc_census_merge.py` - To merge the datasets to get the parliamentary constituency level dataset from the assembly data
2. Run the python script from the dataset directory `add_triplet_urban_flag.py` - To add the urban flag to the dataset based on the population density of the region
3. Run the python script from the training directory `train_election_models.py` - To train the model(s)
4. Run the streamlit command from the dashboard directory `streamlit run streamlit_app.py` - To spin the dashboard in a streamlit app