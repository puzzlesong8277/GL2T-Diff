import synapseclient
syn = synapseclient.Synapse()
syn.login(authToken="YOUR_TOKEN_HERE")
# Obtain a pointer and download the data
syn60086071 = syn.get(entity='syn60086071', version=2 )
# Get the path to the local copy of the data file
filepath = syn60086071.path