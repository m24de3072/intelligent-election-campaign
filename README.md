# Steps to process the dataset, train the model(s) and then view the results in a streamlit app.

1. Run the python script `dataset\create_ac_pc_census_merge.py` - To merge the datasets to get the parliamentary constituency level dataset from the assembly data
2. Run the python script `dataset\add_triplet_urban_flag.py` - To add the urban flag to the dataset based on the population density of the region
3. Run the python script `training\train_election_models.py` - To train the model(s)
4. Run the python script `dashboard\streamlit_app.py` - To spin the dashboard in a streamlit app