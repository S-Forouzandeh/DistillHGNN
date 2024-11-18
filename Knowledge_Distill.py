import os
import pandas as pd
import networkx as nx
import numpy as np
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import softmax, cross_entropy, kl_div
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split  # Import this for splitting datasets
import time

def process_data(folder_path):
    genre_mapping = {}
    genre_id_mapping = {}

    # Read genres from the .... file
    df_genres = pd.read_excel(os.path.join(folder_path, '...'), usecols=['movieID', 'category'])

    # Create a mapping of genre names to unique integer labels
    unique_genres = df_genres['category'].unique()
    genre_id_mapping = {genre: idx for idx, genre in enumerate(sorted(unique_genres))}

    # Replace genre names with integer labels in the DataFrame
    df_genres['category'] = df_genres['category'].map(genre_id_mapping)

    # Create a mapping of movieID to category
    genre_mapping = df_genres.set_index('movieID')['category'].to_dict()

    # Create a DataFrame with only category for MLP training
    ground_truth_ratings = pd.DataFrame(list(genre_mapping.items()), columns=['movieID', 'category'])

    print("Ground truth ratings with genre labels:")
    print(ground_truth_ratings.head())

    return ground_truth_ratings

def create_heterogeneous_graph(folder_path):
    # Create an empty graph
    G = nx.Graph()
    # Create dictionaries to store the number of nodes for each node type
    node_counts = {'userID': 0, 'movieID': 0, 'directorID': 0, 'actorID': 0}

    # Create a dictionary to store mapping between nodes and their attributes
    node_attributes = {}
    # Create a dictionary to store mapping between edges and their weights
    edge_weights = {}

    # Create dictionaries to store the number of nodes and edges for each type of relationship
    relationship_counts = {}

    # Create a dictionary to map each file to its corresponding columns
    file_columns = {
        'user_movies': ['userID', 'movieID', 'rating'],
        'movie_directors': ['movieID', 'directorID'],
        'movie_actors': ['movieID', 'actorID']
    }

    # Iterate through the files and read them to populate the graph
    for file_name, columns in file_columns.items():
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            # Read the Excel file into a pandas DataFrame
            df = pd.read_excel(file_path, usecols=columns)

            # Add nodes and edges to the graph based on the file's content
            if 'userID' in columns:
                for _, row in df.iterrows():
                    user_node = f"userID:{row['userID']}"
                    movie_node = f"movieID:{row['movieID']}"
                    rating = row['rating']

                    # Add nodes only if they don't exist
                    if user_node not in G:
                        G.add_node(user_node, type='userID')
                        node_counts['userID'] += 1

                    if movie_node not in G:
                        G.add_node(movie_node, type='movieID')
                        node_counts['movieID'] += 1

                    G.add_edge(user_node, movie_node, weight=rating)

            if 'directorID' in columns:
                for _, row in df.iterrows():
                    movie_node = f"movieID:{row['movieID']}"
                    director_node = f"directorID:{row['directorID']}"

                    # Add nodes only if they don't exist
                    if movie_node not in G:
                        G.add_node(movie_node, type='movieID')
                        node_counts['movieID'] += 1

                    if director_node not in G:
                        G.add_node(director_node, type='directorID')
                        node_counts['directorID'] += 1

                    G.add_edge(movie_node, director_node)

            if 'actorID' in columns:
                for _, row in df.iterrows():
                    movie_node = f"movieID:{row['movieID']}"
                    actor_node = f"actorID:{row['actorID']}"

                    # Add nodes only if they don't exist
                    if movie_node not in G:
                        G.add_node(movie_node, type='movieID')
                        node_counts['movieID'] += 1

                    if actor_node not in G:
                        G.add_node(actor_node, type='actorID')
                        node_counts['actorID'] += 1

                    G.add_edge(movie_node, actor_node)
    return G


#****************************************************************************************
#----------------------------------- HyperGNN-----------------------------
#****************************************************************************************

def hypergraph_MU(folder_path):

    # Create an empty hypergraph
    hyper_MU = {}
    relationship_counts = {}

    # Create a dictionary to store mapping between nodes and their attributes
    att_MU = {}
    # Create a dictionary to store mapping between edges and their weights
    edge_weights = {}

    # Create a dictionary to map the 'user_movies' file to its corresponding columns
    file_columns = {
        'user_movies': ['userID', 'movieID', 'rating'],
    }

    # Iterate through the files and read them to populate the hypergraph
    for file_name, columns in file_columns.items():
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            # Read the Excel file into a pandas DataFrame
            df = pd.read_excel(file_path, usecols=columns)

            # Update the hypergraph and relationship counts based on the file's content
            for _, row in df.iterrows():
                movie_node = f"movieID:{row['movieID']}"
                user_node = f"userID:{str(row['userID'])}"
                rating = row['rating']

                # Add the movie node to the hypergraph if it doesn't exist
                if movie_node not in hyper_MU:
                    hyper_MU[movie_node] = []

                # Add the user node to the hypergraph if it doesn't exist
                if user_node not in hyper_MU:
                    hyper_MU[user_node] = []

                # Add the user node to the movie hyperedge
                hyper_MU[movie_node].append(user_node)

                # Set the type attribute in att_MU
                att_MU[user_node] = {'type': 'userID'}
                att_MU[movie_node] = {'type': 'movieID'}

                edge_weights[(movie_node, user_node)] = rating

                # Count nodes and edges for the userID-movieID relationship
                relationship = 'userID-movieID'
                relationship_counts[relationship] = relationship_counts.get(relationship, {'nodes': 0, 'edges': 0})
                relationship_counts[relationship]['nodes'] += 2  # Two nodes (movie and user)
                relationship_counts[relationship]['edges'] += 1

    # Filter out hyperedges with empty relationships
    hyper_MU = {k: v for k, v in hyper_MU.items() if v}
    
    # Count the number of edges
    num_edges = sum(len(nodes) for nodes in hyper_MU.values())

    return hyper_MU, att_MU

def hypergraph_MD(folder_path):
 
    # Create an empty hyper_MD
    hyper_MD = {}
    relationship_counts_MD = {}

    # Create a dictionary to store mapping between nodes and their attributes
    att_MD = {}
    
    # Create a dictionary to map the 'director_movies' file to its corresponding columns
    file_columns = {
        'movie_directors': ['movieID', 'directorID'],
    }

    # Iterate through the files and read them to populate the hyper_MD
    for file_name, columns in file_columns.items():
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            # Read the Excel file into a pandas DataFrame
            df = pd.read_excel(file_path, usecols=columns)

            # Update the hyper_MD and relationship_counts based on the file's content
            for _, row in df.iterrows():
                movie_node = f"movieID:{row['movieID']}"
                director_node = f"directorID:{str(row['directorID'])}"

                # Add the movie node to the hypergraph if it doesn't exist
                if movie_node not in hyper_MD:
                    hyper_MD[movie_node] = []

                # Add the director node to the hyper_MD if it doesn't exist
                if director_node not in hyper_MD:
                    hyper_MD[director_node] = []

                # Add the director node to the movie hyperedge
                hyper_MD[movie_node].append(director_node)

                # Set the type attribute in att_MD
                att_MD[director_node] = {'type': 'directorID'}
                att_MD[movie_node] = {'type': 'movieID'}

                # Count nodes and edges for the directorID-movieID relationship
                relationship = 'directorID-movieID'
                relationship_counts_MD[relationship] = relationship_counts_MD.get(relationship, {'nodes': 0, 'edges': 0})
                relationship_counts_MD[relationship]['nodes'] += 2  # Two nodes (movie and director)
                relationship_counts_MD[relationship]['edges'] += 1

    # Filter out hyperedges with empty relationships
    hyper_MD = {k: v for k, v in hyper_MD.items() if v}

    # Count the number of edges
    num_edges = sum(len(nodes) for nodes in hyper_MD.values())

    return hyper_MD, att_MD

def hypergraph_MA(folder_path):

    # Create an empty hyper_MA
    hyper_MA = {}
    relationship_counts_MA = {}

    # Create a dictionary to store mapping between nodes and their attributes
    att_MA = {}
    
    # Create a dictionary to map the 'actor_movies' file to its corresponding columns
    file_columns = {
        'movie_actors': ['movieID', 'actorID'],
    }

    # Iterate through the files and read them to populate the hyper_MA
    for file_name, columns in file_columns.items():
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            # Read the Excel file into a pandas DataFrame
            df = pd.read_excel(file_path, usecols=columns)

            # Update the hyper_MA and relationship_counts based on the file's content
            for _, row in df.iterrows():
                movie_node = f"movieID:{row['movieID']}"
                actor_node = f"actorID:{str(row['actorID'])}"

                # Add the movie node to the hypergraph if it doesn't exist
                if movie_node not in hyper_MA:
                    hyper_MA[movie_node] = []

                # Add the actor node to the hyper_MA if it doesn't exist
                if actor_node not in hyper_MA:
                    hyper_MA[actor_node] = []

                # Add the actor node to the movie hyperedge
                hyper_MA[movie_node].append(actor_node)

                # Set the type attribute in att_MA
                att_MA[actor_node] = {'type': 'actorID'}
                att_MA[movie_node] = {'type': 'movieID'}

                # Count nodes and edges for the actorID-movieID relationship
                relationship = 'actorID-movieID'
                relationship_counts_MA[relationship] = relationship_counts_MA.get(relationship, {'nodes': 0, 'edges': 0})
                relationship_counts_MA[relationship]['nodes'] += 2  # Two nodes (movie and actor)
                relationship_counts_MA[relationship]['edges'] += 1

    # Filter out hyperedges with empty relationships
    hyper_MA = {k: v for k, v in hyper_MA.items() if v}

    # Count the number of edges
    num_edges = sum(len(nodes) for nodes in hyper_MA.values())

    return hyper_MA, att_MA

# Define functions to generate incidence matrices
def generate_incidence_matrices_MU(hyper_MU, att_MU):
    movie_nodes = [node for node in att_MU if att_MU[node]['type'] == 'movieID']
    user_nodes = [node for node in att_MU if att_MU[node]['type'] == 'userID']
    movie_index_map = {movie: i for i, movie in enumerate(movie_nodes)}
    user_index_map = {user: i for i, user in enumerate(user_nodes)}
    num_movies = len(movie_nodes)
    num_users = len(user_nodes)
    incidence_matrix_MU = np.zeros((num_users, num_movies), dtype=float)
    for movie_node, users_connected in hyper_MU.items():
        if movie_node in movie_index_map:
            movie_index = movie_index_map[movie_node]
            for user_node in users_connected:
                if user_node in user_index_map:
                    user_index = user_index_map[user_node]
                    incidence_matrix_MU[user_index, movie_index] = 1
    return incidence_matrix_MU

def generate_incidence_matrices_MD(hyper_MD, att_MD):
    movie_nodes = [node for node in att_MD if att_MD[node]['type'] == 'movieID']
    director_nodes = [node for node in att_MD if att_MD[node]['type'] == 'directorID']
    movie_index_map = {movie: i for i, movie in enumerate(movie_nodes)}
    director_index_map = {director: i for i, director in enumerate(director_nodes)}
    num_movies = len(movie_nodes)
    num_directors = len(director_nodes)
    incidence_matrix_MD = np.zeros((num_directors, num_movies), dtype=float)
    for movie_node, directors_connected in hyper_MD.items():
        if movie_node in movie_index_map:
            movie_index = movie_index_map[movie_node]
            for director_node in directors_connected:
                if director_node in director_index_map:
                    director_index = director_index_map[director_node]
                    incidence_matrix_MD[director_index, movie_index] = 1
    return incidence_matrix_MD

def generate_incidence_matrices_MA(hyper_MA, att_MA):
    movie_nodes = [node for node in att_MA if att_MA[node]['type'] == 'movieID']
    actor_nodes = [node for node in att_MA if att_MA[node]['type'] == 'actorID']
    movie_index_map = {movie: i for i, movie in enumerate(movie_nodes)}
    actor_index_map = {actor: i for i, actor in enumerate(actor_nodes)}
    num_movies = len(movie_nodes)
    num_actors = len(actor_nodes)
    incidence_matrix_MA = np.zeros((num_actors, num_movies), dtype=float)
    for movie_node, actors_connected in hyper_MA.items():
        if movie_node in movie_index_map:
            movie_index = movie_index_map[movie_node]
            for actor_node in actors_connected:
                if actor_node in actor_index_map:
                    actor_index = actor_index_map[actor_node]
                    incidence_matrix_MA[actor_index, movie_index] = 1
    return incidence_matrix_MA

# Function to pad the incidence matrix to a specified size
# Function to pad the incidence matrix to a specified size
def pad_matrix(matrix, target_rows, target_cols):
    current_rows, current_cols = matrix.shape
    # Pad rows if necessary
    row_padding = target_rows - current_rows
    # Pad columns if necessary
    col_padding = target_cols - current_cols
    if row_padding > 0 or col_padding > 0:
        matrix = F.pad(matrix, (0, col_padding, 0, row_padding), "constant", 0)
    return matrix

# Function to compute degree matrices
def compute_degree_matrices(incidence_matrix):
    D_v = torch.diag(torch.sum(incidence_matrix, dim=1))
    D_e = torch.diag(torch.sum(incidence_matrix, dim=0))
    return D_v, D_e

class HypergraphNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HypergraphNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.theta = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.theta_final = nn.Parameter(torch.Tensor(hidden_dim, output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.theta_final)

    def forward(self, incidence_matrix, features):
        D_v_inv_sqrt, D_e_inv = compute_degree_matrices(incidence_matrix)
        X = torch.matmul(D_v_inv_sqrt, features)
        X_hyperedge = torch.matmul(incidence_matrix.T, X)
        X_hyperedge = torch.matmul(D_e_inv, X_hyperedge)
        X = torch.matmul(incidence_matrix, X_hyperedge)
        X = torch.matmul(D_v_inv_sqrt, X)
        return X
    
    def generate_soft_labels(self, incidence_matrix, features):
        """Generate soft labels (probabilities) from the model."""
        with torch.no_grad():
            logits = self.forward(incidence_matrix, features)  # Provide both incidence_matrix and features
            soft_labels = F.softmax(logits, dim=1)  # Convert logits to probabilities
        return soft_labels

class MLPHyper(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPHyper, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)      # First hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)     # Second hidden layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)     # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU after first layer
        x = F.relu(self.fc2(x))  # Apply ReLU after second layer
        x = self.fc3(x)          # Final output layer (logits)
        return x

    def predict(self, x):
        """Generate discrete class predictions based on the soft labels."""
        with torch.no_grad():
            logits = self.forward(x)  # Get raw output logits
            _, predicted = torch.max(logits, 1)  # Get predicted class labels
        return predicted

def generate_hyperGNN_embeddings(hypergraph_model, hyper_MU, hyper_MD, hyper_MA, att_MU, att_MD, att_MA, features):
    incidence_matrix_MU = generate_incidence_matrices_MU(hyper_MU, att_MU)
    incidence_matrix_MD = generate_incidence_matrices_MD(hyper_MD, att_MD)
    incidence_matrix_MA = generate_incidence_matrices_MA(hyper_MA, att_MA)

    # Convert to tensors
    incidence_matrix_MU = torch.tensor(incidence_matrix_MU, dtype=torch.float32)
    incidence_matrix_MD = torch.tensor(incidence_matrix_MD, dtype=torch.float32)
    incidence_matrix_MA = torch.tensor(incidence_matrix_MA, dtype=torch.float32)

    max_num_users = max(incidence_matrix_MU.shape[0], incidence_matrix_MD.shape[0], incidence_matrix_MA.shape[0])

    # Pad incidence matrices and features
    padded_incidence_matrix_MU = pad_matrix(incidence_matrix_MU, max_num_users, incidence_matrix_MU.shape[1])
    padded_incidence_matrix_MD = pad_matrix(incidence_matrix_MD, max_num_users, incidence_matrix_MD.shape[1])
    padded_incidence_matrix_MA = pad_matrix(incidence_matrix_MA, max_num_users, incidence_matrix_MA.shape[1])

    features_MU = pad_matrix(features[:incidence_matrix_MU.shape[0], :], max_num_users, features.shape[1])
    features_MD = pad_matrix(features[:incidence_matrix_MD.shape[0], :], max_num_users, features.shape[1])
    features_MA = pad_matrix(features[:incidence_matrix_MA.shape[0], :], max_num_users, features.shape[1])

    # Generate embeddings using the passed hypergraph model
    embeddings_MU = hypergraph_model(padded_incidence_matrix_MU, features_MU)
    embeddings_MD = hypergraph_model(padded_incidence_matrix_MD, features_MD)
    embeddings_MA = hypergraph_model(padded_incidence_matrix_MA, features_MA)

    HyperGNN_embeddings = embeddings_MU + embeddings_MD + embeddings_MA
    print("HyperGNN Embedding Shape:", HyperGNN_embeddings.shape)
    print("HyperGNN Embedding:", HyperGNN_embeddings)

    return HyperGNN_embeddings

def prepare_data_split(filtered_embeddings, genre_labels):
    if filtered_embeddings.size(0) == 0 or genre_labels.size(0) == 0:
        raise ValueError("Filtered embeddings or genre labels are empty!")
    
    train_embeddings, test_embeddings, train_labels, test_labels = train_test_split(
        filtered_embeddings.detach().numpy(),
        genre_labels.numpy(),
        test_size=0.3,
        random_state=42
    )

    train_embeddings = torch.tensor(train_embeddings, dtype=torch.float32)
    test_embeddings = torch.tensor(test_embeddings, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    return train_embeddings, test_embeddings, train_labels, test_labels

def evaluate_HypergraphModel(model, test_embeddings, test_labels):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        test_outputs = model(test_embeddings)  # Get predictions
        _, predicted = torch.max(test_outputs, 1)  # Get class predictions
        # Calculate accuracy if test_labels are provided
        accuracy = (predicted == test_labels).sum().item() / test_labels.size(0) if test_labels is not None else None
        
    return predicted, accuracy

def prepare_data_split(filtered_embeddings, genre_labels):
    if filtered_embeddings.size(0) == 0 or genre_labels.size == 0:
        raise ValueError("Filtered embeddings or genre labels are empty!")

    train_embeddings, test_embeddings, train_labels, test_labels = train_test_split(
        filtered_embeddings.detach().numpy(),
        genre_labels,  # Pass directly without .to_numpy()
        test_size=0.3,
        random_state=42
    )

    # Convert back to tensors
    train_embeddings = torch.tensor(train_embeddings, dtype=torch.float32)
    test_embeddings = torch.tensor(test_embeddings, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    return train_embeddings, test_embeddings, train_labels, test_labels

#****************************************************************************************
#----------------------------------- TinyGCN-----------------------------
#****************************************************************************************

# Function to create separate subgraphs for each relation
def create_relation_graphs(folder_path):
    G_user_movie = nx.Graph()  # Graph for user-movie relations
    G_movie_director = nx.Graph()  # Graph for movie-director relations
    G_movie_actor = nx.Graph()  # Graph for movie-actor relations
    
    file_columns = {
        'user_movies': ['userID', 'movieID', 'rating'],
        'movie_directors': ['movieID', 'directorID'],
        'movie_actors': ['movieID', 'actorID']
    }

    for file_name, columns in file_columns.items():
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            df = pd.read_excel(file_path, usecols=columns)

            if 'userID' in columns:  # User-movie relations
                for _, row in df.iterrows():
                    user_node = f"userID:{row['userID']}"
                    movie_node = f"movieID:{row['movieID']}"
                    rating = row['rating']

                    G_user_movie.add_edge(user_node, movie_node, weight=rating)

            if 'directorID' in columns:  # Movie-director relations
                for _, row in df.iterrows():
                    movie_node = f"movieID:{row['movieID']}"
                    director_node = f"directorID:{row['directorID']}"
                    G_movie_director.add_edge(movie_node, director_node)

            if 'actorID' in columns:  # Movie-actor relations
                for _, row in df.iterrows():
                    movie_node = f"movieID:{row['movieID']}"
                    actor_node = f"actorID:{row['actorID']}"
                    G_movie_actor.add_edge(movie_node, actor_node)

    return G_user_movie, G_movie_director, G_movie_actor

# Function to convert graph into adjacency matrix
def graph_to_adjacency_matrix(G):
    adj_matrix = nx.to_numpy_array(G)
    return torch.tensor(adj_matrix, dtype=torch.float32)

# Function to pad smaller adjacency matrices to match the largest one
def pad_adj_matrices(adjs):
    max_size = max([adj.size(0) for adj in adjs])  # Find the largest size

    padded_adjs = []
    for adj in adjs:
        size_diff = max_size - adj.size(0)
        if size_diff > 0:
            # Pad the matrix with zeros to make it square of size max_size
            adj = F.pad(adj, (0, size_diff, 0, size_diff), "constant", 0)
        padded_adjs.append(adj)

    return padded_adjs

class LightGNNLayer(nn.Module):
    def __init__(self):
        super(LightGNNLayer, self).__init__()

    def forward(self, adjacency_matrix, features):
        # LightGCN aggregation step: A * features (no transformation or activation)
        h = torch.mm(adjacency_matrix, features)
        return h  # No non-linearity, returning aggregated embeddings directly

# LightGCN model with one layer
class LightGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, gnn_output_dim):
        super(LightGNN, self).__init__()
        self.layer = LightGNNLayer()  # Single layer LightGCN
        
        # Define additional layers if needed based on the dimensions provided
        self.fc = nn.Linear(input_dim, gnn_output_dim)  # Example if you want an additional layer
    
    def forward(self, adj, features):
        h = self.layer(adj, features)  # Single LightGCN layer
        h = self.fc(h)  # Apply additional transformation if needed
        return h  # Output node embeddings

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, hyper_embeddings, light_embeddings, labels):
        # Normalize the embeddings
        hyper_embeddings = F.normalize(hyper_embeddings, dim=1)
        light_embeddings = F.normalize(light_embeddings, dim=1)

        # Compute similarity matrices
        hyper_similarity = torch.mm(hyper_embeddings, hyper_embeddings.T) / self.temperature
        light_similarity = torch.mm(light_embeddings, light_embeddings.T) / self.temperature

        # Combine the similarities into one similarity matrix
        combined_similarity = torch.cat([hyper_similarity, light_similarity], dim=0)
        combined_size = combined_similarity.size(0)  # Should be 2 * batch_size

        # Create labels for positive pairs
        batch_size = labels.size(0)
        labels = labels.unsqueeze(1)  # Make labels a column vector
        
        # Create a mask for positive pairs
        mask = torch.eq(labels, labels.T).float()  # Positive pairs mask
        mask = torch.cat([mask, mask], dim=0)  # Adjust mask for combined size
        mask.fill_diagonal_(0)  # Remove diagonal for negative sampling
        
        # Ensure the mask matches the combined similarity size
        mask = mask[:combined_size, :combined_size]  # Ensure correct size
        
        # Compute logits
        logits = combined_similarity - torch.max(combined_similarity, dim=1, keepdim=True)[0]

        # Calculate probabilities
        exp_logits = torch.exp(logits) * mask  # Apply mask to logits
        exp_logits_sum = exp_logits.sum(dim=1, keepdim=True) + 1e-10  # Sum for normalization

        # Compute the log probability
        log_prob = exp_logits / exp_logits_sum

        # Compute the InfoNCE loss
        loss = -torch.log(log_prob + 1e-10)  # Add small value for numerical stability
        return loss.mean()

def combine_embeddings(hyper_embeddings, light_embeddings):
    # Weight the embeddings (you can adjust the weights as needed)
    weight_hyper = 0.5
    weight_light = 0.2

    # Ensure both embeddings have the same number of features
    if light_embeddings.shape[1] > hyper_embeddings.shape[1]:
        light_embeddings_reduced = light_embeddings[:, :hyper_embeddings.shape[1]]
    else:
        light_embeddings_reduced = F.adaptive_avg_pool2d(light_embeddings.view(1, -1, light_embeddings.shape[1]), 
                                                         (1, hyper_embeddings.shape[1])).view(-1, hyper_embeddings.shape[1])
    
    combined_embeddings = weight_hyper * hyper_embeddings + weight_light * light_embeddings_reduced
    return combined_embeddings

def encode_labels(genre_labels):
    """Encodes labels to ensure they are within the valid range."""
    unique_labels = np.unique(genre_labels)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = np.array([label_to_index[label] for label in genre_labels])
    return encoded_labels, len(unique_labels)

def combined_train_model(hyper_embeddings, light_embeddings, genre_labels, num_classes, num_epochs=500, learning_rate=0.001):
    # Initialize models
    mlp_hypergraph_model = MLPHyper(input_dim=hyper_embeddings.size(1), hidden_dim=64, output_dim=num_classes).to(hyper_embeddings.device)
    light_gnn_model = LightGNN(input_dim=hyper_embeddings.size(1), hidden_dim=64, gnn_output_dim=16).to(hyper_embeddings.device)

    # Loss functions
    cross_entropy_loss_fn = nn.CrossEntropyLoss()  # For HypergraphNN MLP
    info_nce_loss_fn = InfoNCELoss(temperature=0.1)  # For LightGNN contrastive learning

    # Optimizers
    optimizer = torch.optim.Adam(list(mlp_hypergraph_model.parameters()) + list(light_gnn_model.parameters()), lr=learning_rate)

    # Convert genre_labels to tensor
    genre_labels = torch.tensor(genre_labels).to(hyper_embeddings.device).clone().detach()

    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Clear gradients for the combined optimizer

        # --- HypergraphNN MLP Training ---
        outputs_hyper = mlp_hypergraph_model(hyper_embeddings)  # Get predictions from HypergraphNN's MLP

        # Cross-entropy loss (classification loss for HypergraphNN)
        loss_hyper = cross_entropy_loss_fn(outputs_hyper, genre_labels)

        # --- LightGNN Training ---
        combined_embeddings = combine_embeddings(hyper_embeddings, light_embeddings)  # Combine embeddings for contrastive learning

        # InfoNCE loss (contrastive learning loss for LightGNN)
        loss_light = info_nce_loss_fn(hyper_embeddings, combined_embeddings, genre_labels)

        # --- Combined Loss ---
        total_loss = loss_hyper + loss_light  # Sum of both losses

        # Backpropagation for both models
        total_loss.backward(retain_graph=True)  # Compute gradients for the total loss

        # Optimizers step for both models
        optimizer.step()  # Update parameters for both models

        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Total Loss: {total_loss.item():.4f}, Cross-Entropy Loss: {loss_hyper.item():.4f}, InfoNCE Loss: {loss_light.item():.4f}')

    # Return both models (if needed later for inference)
    return mlp_hypergraph_model, light_gnn_model

# Function to generate embeddings from multiple adjacency matrices and concatenate them
def generate_LightGNN_embeddings(folder_path, features):
    G_user_movie, G_movie_director, G_movie_actor = create_relation_graphs(folder_path)
    
    # Convert graphs to adjacency matrices
    adj_user_movie = graph_to_adjacency_matrix(G_user_movie)
    adj_movie_director = graph_to_adjacency_matrix(G_movie_director)
    adj_movie_actor = graph_to_adjacency_matrix(G_movie_actor)

    # Pad adjacency matrices
    padded_adjs = pad_adj_matrices([adj_user_movie, adj_movie_director, adj_movie_actor])

    # Create LightGNN model
    input_dim = features.shape[1]
    hidden_dim = 64
    gnn_output_dim = 16
    gnn_model = LightGNN(input_dim, hidden_dim, gnn_output_dim)

    # Generate embeddings for each relation
    embeddings = []
    for adj in padded_adjs:
        emb = gnn_model(adj, features)
        embeddings.append(emb)

    # Concatenate all embeddings
    LightGNN_embeddings = torch.cat(embeddings, dim=1)  # Concatenate along the feature dimension

    return LightGNN_embeddings

#---------------Knowledge Distillation ----------------------
# Define the MLP model
class MLPDistll(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPDistll, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Final output layer
        return x

# Generate embeddings from LightGNN
def generate_LightGNN_embeddings(folder_path, features):
    start_time = time.time()  # Start timing

    G_user_movie, G_movie_director, G_movie_actor = create_relation_graphs(folder_path)
    adj_user_movie = graph_to_adjacency_matrix(G_user_movie)
    adj_movie_director = graph_to_adjacency_matrix(G_movie_director)
    adj_movie_actor = graph_to_adjacency_matrix(G_movie_actor)

    padded_adjs = pad_adj_matrices([adj_user_movie, adj_movie_director, adj_movie_actor])
    input_dim = features.shape[1]
    hidden_dim = 64
    gnn_output_dim = 16

    gnn_model = LightGNN(input_dim, hidden_dim, gnn_output_dim)

    embeddings = []
    for adj in padded_adjs:
        emb = gnn_model(adj, features)
        embeddings.append(emb)

    LightGNN_embeddings = torch.cat(embeddings, dim=1)    
    end_time = time.time()  # End timing
    print(f"Embedding generation time: {end_time - start_time:.4f} seconds")
    
    return LightGNN_embeddings

# Knowledge distillation loss function
def distillation_loss(predictions, hard_labels, soft_labels, temperature=1.0, alpha=0.5):
    ce_loss = F.cross_entropy(predictions, hard_labels)

    log_preds = F.log_softmax(predictions / temperature, dim=1)
    soft_labels = F.softmax(soft_labels / temperature, dim=1)

    # Ensure shapes match before calculating KL divergence
    if log_preds.size(1) != soft_labels.size(1):
        # Adjust soft_labels to match predictions
        soft_labels = soft_labels[:, :log_preds.size(1)]  # Truncate soft_labels if they are larger

    kl_loss = F.kl_div(log_preds, soft_labels, reduction='batchmean') * (temperature ** 2)

    return alpha * ce_loss + (1 - alpha) * kl_loss

def process_data(folder_path):
    genre_mapping = {}
    genre_id_mapping = {}

    # Read genres from the movie_genres file
    df_genres = pd.read_excel(os.path.join(folder_path, '.....'), usecols=['movieID', 'Category'])

    # Create a mapping of genre names to unique integer labels
    unique_genres = df_genres['category'].unique()
    genre_id_mapping = {genre: idx for idx, genre in enumerate(sorted(unique_genres))}

    # Replace genre names with integer labels in the DataFrame
    df_genres['category'] = df_genres['category'].map(genre_id_mapping)

    # Create a mapping of movieID to category
    genre_mapping = df_genres.set_index('movieID')['category'].to_dict()

    # Create a DataFrame with only category for MLP training
    ground_truth_ratings = pd.DataFrame(list(genre_mapping.items()), columns=['movieID', 'category'])

    return ground_truth_ratings

def evaluate_GNNmodel_with_distillation(model, embeddings, labels, teacher_soft_labels, temperature=1.0, alpha=0.5):
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        
        outputs = model(embeddings)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).sum().item() / labels.size(0)
        loss_value = distillation_loss(outputs, labels, teacher_soft_labels, temperature, alpha)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
    return predicted, accuracy, loss_value, inference_time

def prepare_data_split(filtered_embeddings, genre_labels, random_seed):
    if filtered_embeddings.size(0) != genre_labels.size(0):
        raise ValueError("Number of samples in embeddings and labels must match.")

    # Split into train and temp (which will be further split into validation and test)
    train_embeddings, temp_embeddings, train_labels, temp_labels = train_test_split(
        filtered_embeddings.detach().numpy(),
        genre_labels.numpy(),
        test_size=0.3,  # 30% for validation + test
        random_state=random_seed
    )

    # Split the temp set into validation and test sets
    val_size = 0.5  # 50% of temp goes to validation and 50% goes to testing
    val_embeddings, test_embeddings, val_labels, test_labels = train_test_split(
        temp_embeddings,
        temp_labels,
        test_size=val_size,
        random_state=random_seed
    )

    train_embeddings = torch.tensor(train_embeddings, dtype=torch.float32)
    val_embeddings = torch.tensor(val_embeddings, dtype=torch.float32)
    test_embeddings = torch.tensor(test_embeddings, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    # Count the number of nodes (samples) in each split
    train_nodes = train_labels.size(0)
    val_nodes = val_labels.size(0)
    test_nodes = test_labels.size(0)
    
    print(f"Training Nodes: {train_nodes}")
    print(f"Validation Nodes: {val_nodes}")
    print(f"Testing Nodes: {test_nodes}")
    
    return train_embeddings, val_embeddings, test_embeddings, train_labels, val_labels, test_labels

def run_production_evaluation(light_embeddings, genre_labels, num_classes):
    transductive_accuracies = []
    transductive_losses = []
    transductive_inference_times = []

    inductive_accuracies = []
    inductive_losses = []
    inductive_inference_times = []

    production_accuracies = []
    production_losses = []
    production_inference_times = []

    for seed in range(5):
        train_embeddings, val_embeddings, test_embeddings, train_labels, val_labels, test_labels = prepare_data_split(
            light_embeddings, genre_labels, seed
        )
        
        teacher_train_soft_labels = torch.softmax(train_embeddings / 1.0, dim=1)
        teacher_val_soft_labels = torch.softmax(val_embeddings / 1.0, dim=1)
        teacher_test_soft_labels = torch.softmax(test_embeddings / 1.0, dim=1)

        input_dim = light_embeddings.shape[1]
        hidden_dim = 64
        output_dim = num_classes
        dis_mlp_model = MLPDistll(input_dim, hidden_dim, output_dim)

        # Evaluate on validation set
        val_predictions, val_accuracy, val_loss, val_inference_time = evaluate_GNNmodel_with_distillation(
            dis_mlp_model, val_embeddings, val_labels, teacher_val_soft_labels, temperature=1.0, alpha=0.5
        )
        
        print(f"Validation Accuracy for seed {seed}: {val_accuracy * 100:.2f}%")

        # Evaluate on test set
        trans_predictions, trans_accuracy, trans_loss, trans_inference_time = evaluate_GNNmodel_with_distillation(
            dis_mlp_model, test_embeddings, test_labels, teacher_test_soft_labels, temperature=1.0, alpha=0.5
        )
        
        transductive_accuracies.append(trans_accuracy)
        transductive_losses.append(trans_loss)
        transductive_inference_times.append(trans_inference_time)
        
        print(f"Inference time for seed {seed} (Transductive): {trans_inference_time:.4f} seconds")

        # Inductive setting
        inductive_train_embeddings, inductive_val_embeddings, inductive_test_embeddings, inductive_train_labels, inductive_val_labels, inductive_test_labels = prepare_data_split(
            light_embeddings, genre_labels, seed
        )
        
        teacher_inductive_train_soft_labels = torch.softmax(inductive_train_embeddings / 1.0, dim=1)
        teacher_inductive_test_soft_labels = torch.softmax(inductive_test_embeddings / 1.0, dim=1)
        
        # Evaluate the MLP model for inductive setting
        inductive_predictions, inductive_accuracy, inductive_loss, inductive_inference_time = evaluate_GNNmodel_with_distillation(
            dis_mlp_model, inductive_test_embeddings, inductive_test_labels, teacher_inductive_test_soft_labels, temperature=1.0, alpha=0.5
        )
        
        inductive_accuracies.append(inductive_accuracy)
        inductive_losses.append(inductive_loss)
        inductive_inference_times.append(inductive_inference_time)

        print(f"Inference time for seed {seed} (Inductive): {inductive_inference_time:.4f} seconds")

        # Production setting
        combined_predictions = (trans_predictions + inductive_predictions) / 2
        combined_loss = (trans_loss + inductive_loss) / 2
        combined_accuracy = (trans_accuracy + inductive_accuracy) / 2
        combined_inference_time = (trans_inference_time + inductive_inference_time) / 2
        
        production_accuracies.append(combined_accuracy)
        production_losses.append(combined_loss)
        production_inference_times.append(combined_inference_time)

        print(f"Inference time for seed {seed} (Production): {combined_inference_time:.4f} seconds")

    # Calculate average results and standard deviation for all three settings
    avg_trans_accuracy = np.mean(transductive_accuracies)
    std_trans_accuracy = np.std(transductive_accuracies)
    avg_trans_loss = np.mean(transductive_losses)
    std_trans_loss = np.std(transductive_losses)
    avg_trans_inference_time = np.mean(transductive_inference_times)
    std_trans_inference_time = np.std(transductive_inference_times)
    
    avg_inductive_accuracy = np.mean(inductive_accuracies)
    std_inductive_accuracy = np.std(inductive_accuracies)
    avg_inductive_loss = np.mean(inductive_losses)
    std_inductive_loss = np.std(inductive_losses)
    avg_inductive_inference_time = np.mean(inductive_inference_times)
    std_inductive_inference_time = np.std(inductive_inference_times)
    
    avg_prod_accuracy = np.mean(production_accuracies)
    std_prod_accuracy = np.std(production_accuracies)
    avg_prod_loss = np.mean(production_losses)
    std_prod_loss = np.std(production_losses)
    avg_prod_inference_time = np.mean(production_inference_times)
    std_prod_inference_time = np.std(production_inference_times)
    
    # Output average results and standard deviation for all settings
    print(f"Average Inference Time (Transductive): {avg_trans_inference_time:.4f} seconds (± {std_trans_inference_time:.4f})")  
    print(f"Average Inference Time (Inductive): {avg_inductive_inference_time:.4f} seconds (± {std_inductive_inference_time:.4f})")
    print(f"Average Inference Time (Production): {avg_prod_inference_time:.4f} seconds (± {std_prod_inference_time:.4f})")
    print(f"Average Test Accuracy (Transductive): {avg_trans_accuracy * 100:.2f}% (± {std_trans_accuracy * 100:.2f}%)")  
    print(f"Average Test Accuracy (Inductive): {avg_inductive_accuracy * 100:.2f}% (± {std_inductive_accuracy * 100:.2f}%)")
    print(f"Average Test Accuracy (Production): {avg_prod_accuracy * 100:.2f}% (± {std_prod_accuracy * 100:.2f}%)")

def main():
    folder_path = '....'
    ground_truth_ratings = process_data(folder_path)

    # Extract genre labels
    genre_labels = ground_truth_ratings['category'].values

    # Encode labels
    genre_labels, num_classes = encode_labels(genre_labels)

    # Generate hypergraph structures and attributes
    hyper_MU, att_MU = hypergraph_MU(folder_path)
    hyper_MD, att_MD = hypergraph_MD(folder_path)
    hyper_MA, att_MA = hypergraph_MA(folder_path)

    # Initialize features and model parameters
    num_nodes = max(len(att_MU), len(att_MD), len(att_MA))
    feature_dim = 32
    features = torch.randn(num_nodes, feature_dim)

    # Generate LightGNN embeddings and time the process
    start_embedding_time = time.time()
    light_embeddings = generate_LightGNN_embeddings(folder_path, features)
    end_embedding_time = time.time()
    embedding_generation_time = end_embedding_time - start_embedding_time

    # Print the embedding generation time once
    print(f"Embedding generation time: {embedding_generation_time:.4f} seconds")

    # Filter or adjust embeddings and labels to match in size
    if len(genre_labels) < len(light_embeddings):
        light_embeddings = light_embeddings[:len(genre_labels)]
    elif len(genre_labels) > len(light_embeddings):
        genre_labels = genre_labels[:len(light_embeddings)]

    # Convert genre_labels to tensor with long type
    genre_labels = torch.tensor(genre_labels, dtype=torch.int64).to(light_embeddings.device)

    # Run production evaluation (transductive + inductive + production)
    run_production_evaluation(light_embeddings, genre_labels, num_classes)

if __name__ == "__main__":
    main()
