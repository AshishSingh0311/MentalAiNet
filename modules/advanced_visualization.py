"""
Advanced visualization module for MH-Net.

This module provides advanced visualization capabilities for multimodal data,
brain imaging, and network analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from io import BytesIO
import base64
import time
import json
from typing import Dict, List, Optional, Union, Any, Tuple


class Brain3DVisualizer:
    """Class for 3D brain visualization."""
    
    def __init__(self, width=800, height=600):
        """
        Initialize the brain visualizer.
        
        Args:
            width (int): Width of the visualization
            height (int): Height of the visualization
        """
        self.width = width
        self.height = height
        self.coordinates = None
        self.values = None
        self.region_names = None
    
    def load_coordinates(self, coordinates: np.ndarray, region_names: Optional[List[str]] = None):
        """
        Load brain region coordinates.
        
        Args:
            coordinates (np.ndarray): Array of coordinates with shape (n_regions, 3)
            region_names (list, optional): List of region names
            
        Returns:
            self: For method chaining
        """
        self.coordinates = coordinates
        
        if region_names is None:
            # Generate default region names
            self.region_names = [f"Region {i}" for i in range(len(coordinates))]
        else:
            self.region_names = region_names
        
        return self
    
    def load_values(self, values: np.ndarray):
        """
        Load values for brain regions.
        
        Args:
            values (np.ndarray): Array of values with shape (n_regions,)
            
        Returns:
            self: For method chaining
        """
        self.values = values
        return self
    
    def generate_demo_data(self, n_regions: int = 100):
        """
        Generate demo data for visualization.
        
        Args:
            n_regions (int): Number of brain regions to generate
            
        Returns:
            self: For method chaining
        """
        # Generate coordinates in a roughly brain-shaped pattern
        theta = np.random.uniform(0, 2*np.pi, n_regions)
        phi = np.random.uniform(0, np.pi, n_regions)
        
        # Focus more points in certain brain region shapes
        # Adjust radius to create brain-like shape, more concentrated in certain areas
        r = 0.8 + 0.2 * np.random.randn(n_regions)
        r = np.where(phi < np.pi/2, r * (1.2 - 0.4 * np.cos(phi)), r)
        
        # Convert to cartesian coordinates
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        # Scale to make more brain-like
        x = 1.3 * x
        y = 1.0 * y
        z = 1.0 * z
        
        self.coordinates = np.column_stack((x, y, z))
        
        # Generate values for brain regions (e.g., activation values)
        self.values = np.abs(np.random.randn(n_regions))
        
        # Simulate a few hotspot regions
        hotspot_indices = np.random.choice(n_regions, 5, replace=False)
        self.values[hotspot_indices] *= 3
        
        # Generate region names
        brain_regions = [
            "Frontal", "Temporal", "Parietal", "Occipital", "Cerebellar", "Hippocampal",
            "Amygdala", "Thalamic", "Insular", "Cingulate", "Striatal", "Precuneus"
        ]
        
        self.region_names = []
        for i in range(n_regions):
            region_base = np.random.choice(brain_regions)
            side = np.random.choice(["Left", "Right"])
            sub_region = np.random.randint(1, 5)
            self.region_names.append(f"{side} {region_base}-{sub_region}")
        
        return self
    
    def create_connectome_data(self, density: float = 0.1):
        """
        Generate a connectome (connectivity between brain regions).
        
        Args:
            density (float): Connection density (0-1)
            
        Returns:
            np.ndarray: Connectivity matrix
        """
        if self.coordinates is None:
            raise ValueError("No coordinates loaded. Call load_coordinates() or generate_demo_data() first.")
        
        n_regions = len(self.coordinates)
        
        # Create distance-based connectivity
        connectivity = np.zeros((n_regions, n_regions))
        
        # Compute pairwise distances
        for i in range(n_regions):
            for j in range(i+1, n_regions):
                # Euclidean distance
                dist = np.linalg.norm(self.coordinates[i] - self.coordinates[j])
                
                # Inverse relationship: closer = stronger connection
                strength = 1.0 / (1.0 + 5.0 * dist)
                
                # Apply random factor for variability
                if np.random.random() < density:
                    connectivity[i, j] = strength
                    connectivity[j, i] = strength  # Symmetric connectivity
        
        return connectivity
    
    def plot_3d_brain(self, title: str = "3D Brain Visualization"):
        """
        Create a 3D brain plot using Plotly.
        
        Args:
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        if self.coordinates is None:
            raise ValueError("No coordinates loaded. Call load_coordinates() or generate_demo_data() first.")
        
        # Create figure
        fig = go.Figure()
        
        # Normalize values for sizing and coloring if available
        if self.values is not None:
            # Normalize to range 0-1
            norm_values = (self.values - np.min(self.values)) / (np.max(self.values) - np.min(self.values))
            
            # Scale for point size (5-25)
            point_sizes = 5 + 20 * norm_values
            
            # Custom colormap (blue to red gradient)
            colors = []
            for v in norm_values:
                if v < 0.33:
                    r, g, b = 0, 0, 255 * (3 * v)
                elif v < 0.67:
                    r, g, b = 255 * (3 * (v - 0.33)), 0, 255 * (1 - 3 * (v - 0.33))
                else:
                    r, g, b = 255, 0, 0
                colors.append(f'rgb({int(r)},{int(g)},{int(b)})')
        else:
            # Default size and color if no values
            point_sizes = 8
            colors = 'rgb(70,130,180)'
        
        # Add brain regions as scatter points
        fig.add_trace(go.Scatter3d(
            x=self.coordinates[:, 0],
            y=self.coordinates[:, 1],
            z=self.coordinates[:, 2],
            mode='markers',
            marker=dict(
                size=point_sizes,
                color=colors,
                opacity=0.8
            ),
            text=self.region_names,
            hoverinfo='text'
        ))
        
        # Set layout
        fig.update_layout(
            title=title,
            width=self.width,
            height=self.height,
            scene=dict(
                xaxis=dict(showticklabels=False, title=''),
                yaxis=dict(showticklabels=False, title=''),
                zaxis=dict(showticklabels=False, title=''),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        return fig
    
    def plot_connectome(self, connectivity: Optional[np.ndarray] = None, threshold: float = 0.1, title: str = "Brain Connectome"):
        """
        Create a 3D brain connectome plot using Plotly.
        
        Args:
            connectivity (np.ndarray, optional): Connectivity matrix
            threshold (float): Threshold for displaying connections
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        if self.coordinates is None:
            raise ValueError("No coordinates loaded. Call load_coordinates() or generate_demo_data() first.")
        
        if connectivity is None:
            # Generate connectivity if not provided
            connectivity = self.create_connectome_data()
        
        # Create figure
        fig = go.Figure()
        
        # Add brain regions as scatter points
        if self.values is not None:
            # Normalize to range 0-1
            norm_values = (self.values - np.min(self.values)) / (np.max(self.values) - np.min(self.values))
            
            # Scale for point size (4-15)
            point_sizes = 4 + 11 * norm_values
            
            # Color mapping
            colors = []
            for v in norm_values:
                if v < 0.33:
                    r, g, b = 0, 0, 255 * (3 * v)
                elif v < 0.67:
                    r, g, b = 255 * (3 * (v - 0.33)), 0, 255 * (1 - 3 * (v - 0.33))
                else:
                    r, g, b = 255, 0, 0
                colors.append(f'rgb({int(r)},{int(g)},{int(b)})')
        else:
            # Default size and color if no values
            point_sizes = 5
            colors = 'rgb(70,130,180)'
        
        fig.add_trace(go.Scatter3d(
            x=self.coordinates[:, 0],
            y=self.coordinates[:, 1],
            z=self.coordinates[:, 2],
            mode='markers',
            marker=dict(
                size=point_sizes,
                color=colors,
                opacity=0.8
            ),
            text=self.region_names,
            hoverinfo='text'
        ))
        
        # Add connections between regions
        n_regions = len(self.coordinates)
        
        # Store edge indices and strengths for lines
        edges_x = []
        edges_y = []
        edges_z = []
        edge_colors = []
        
        for i in range(n_regions):
            for j in range(i+1, n_regions):
                if connectivity[i, j] > threshold:
                    # Extract coordinates for both regions
                    x1, y1, z1 = self.coordinates[i]
                    x2, y2, z2 = self.coordinates[j]
                    
                    # Add line between regions with None to separate lines
                    edges_x.extend([x1, x2, None])
                    edges_y.extend([y1, y2, None])
                    edges_z.extend([z1, z2, None])
                    
                    # Determine edge color based on connection strength
                    strength = connectivity[i, j]
                    edge_colors.extend([strength, strength, strength])
        
        if edges_x:  # Only add trace if there are edges
            fig.add_trace(go.Scatter3d(
                x=edges_x,
                y=edges_y,
                z=edges_z,
                mode='lines',
                line=dict(
                    color=edge_colors,
                    width=2,
                    colorscale='Viridis'
                ),
                hoverinfo='none'
            ))
        
        # Set layout
        fig.update_layout(
            title=title,
            width=self.width,
            height=self.height,
            scene=dict(
                xaxis=dict(showticklabels=False, title=''),
                yaxis=dict(showticklabels=False, title=''),
                zaxis=dict(showticklabels=False, title=''),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        return fig
    
    def plot_region_comparison(self, region_indices=None, title: str = "Region Comparison"):
        """
        Create a bar chart comparing values for selected brain regions.
        
        Args:
            region_indices (list, optional): Indices of regions to compare. If None, top 10 regions by value are used.
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        if self.values is None:
            raise ValueError("No values loaded. Call load_values() or generate_demo_data() first.")
        
        if region_indices is None:
            # Select top 10 regions by value
            region_indices = np.argsort(self.values)[-10:]
        
        # Extract data for selected regions
        selected_regions = [self.region_names[i] for i in region_indices]
        selected_values = [self.values[i] for i in region_indices]
        
        # Sort by value for better visualization
        sorted_indices = np.argsort(selected_values)
        selected_regions = [selected_regions[i] for i in sorted_indices]
        selected_values = [selected_values[i] for i in sorted_indices]
        
        # Create bar chart
        fig = px.bar(
            x=selected_values,
            y=selected_regions,
            orientation='h',
            title=title,
            color=selected_values,
            color_continuous_scale='Viridis',
            labels={"x": "Value", "y": "Region"}
        )
        
        # Adjust layout
        fig.update_layout(
            width=self.width,
            height=self.height
        )
        
        return fig


class NetworkVisualizer:
    """Class for network visualization and analysis."""
    
    def __init__(self, width=800, height=600):
        """
        Initialize the network visualizer.
        
        Args:
            width (int): Width of the visualization
            height (int): Height of the visualization
        """
        self.width = width
        self.height = height
        self.nodes = None
        self.edges = None
        self.node_attributes = None
    
    def load_network(self, nodes: List[Dict], edges: List[Dict], node_attributes: Optional[Dict] = None):
        """
        Load network data.
        
        Args:
            nodes (list): List of node dictionaries with at least 'id' field
            edges (list): List of edge dictionaries with at least 'source' and 'target' fields
            node_attributes (dict, optional): Dictionary mapping node ids to attributes
            
        Returns:
            self: For method chaining
        """
        self.nodes = nodes
        self.edges = edges
        self.node_attributes = node_attributes
        
        return self
    
    def generate_demo_network(self, n_nodes: int = 50, connection_density: float = 0.1):
        """
        Generate a demo network.
        
        Args:
            n_nodes (int): Number of nodes
            connection_density (float): Probability of connection between nodes
            
        Returns:
            self: For method chaining
        """
        # Generate nodes
        self.nodes = []
        node_types = ["symptom", "factor", "outcome", "intervention"]
        domains = ["cognitive", "emotional", "behavioral", "physiological", "social"]
        
        for i in range(n_nodes):
            node_type = np.random.choice(node_types)
            domain = np.random.choice(domains)
            
            # Create more meaningful node names based on type
            if node_type == "symptom":
                symptoms = ["Depressed mood", "Anxiety", "Insomnia", "Rumination", "Fatigue",
                           "Concentration issues", "Irritability", "Appetite changes", "Hopelessness",
                           "Avoidance", "Guilt", "Worthlessness", "Social withdrawal", "Anhedonia"]
                name = np.random.choice(symptoms)
            elif node_type == "factor":
                factors = ["Stress", "Trauma history", "Genetic predisposition", "Social support",
                          "Early life adversity", "Sleep quality", "Diet", "Exercise", "Work environment"]
                name = np.random.choice(factors)
            elif node_type == "outcome":
                outcomes = ["Treatment response", "Functional impairment", "Quality of life",
                           "Relationship quality", "Work performance", "Self-esteem"]
                name = np.random.choice(outcomes)
            else:  # intervention
                interventions = ["CBT", "Medication", "Mindfulness", "Exercise therapy",
                                "Social skills training", "Exposure therapy", "Family therapy"]
                name = np.random.choice(interventions)
            
            self.nodes.append({
                "id": f"n{i}",
                "name": name,
                "type": node_type,
                "domain": domain
            })
        
        # Generate edges
        self.edges = []
        
        # Hierarchical clustering - create more realistic connections
        # Group nodes into clusters
        n_clusters = 5
        clusters = [[] for _ in range(n_clusters)]
        
        for i, node in enumerate(self.nodes):
            cluster_idx = i % n_clusters
            clusters[cluster_idx].append(node)
        
        # Create more connections within clusters, fewer between
        for i, node1 in enumerate(self.nodes):
            node1_cluster = i % n_clusters
            
            for j, node2 in enumerate(self.nodes):
                if i == j:
                    continue  # Skip self-connections
                
                node2_cluster = j % n_clusters
                
                # Higher probability of connection within same cluster
                if node1_cluster == node2_cluster:
                    p_connect = connection_density * 3
                else:
                    p_connect = connection_density * 0.5
                
                if np.random.random() < p_connect:
                    # Create edge with random weight
                    weight = np.round(np.random.uniform(0.1, 1.0), 2)
                    
                    # Adjust weight based on node types (domain knowledge)
                    if node1["type"] == "factor" and node2["type"] == "symptom":
                        weight *= 1.5  # Factors strongly influence symptoms
                    elif node1["type"] == "symptom" and node2["type"] == "outcome":
                        weight *= 1.3  # Symptoms impact outcomes
                    elif node1["type"] == "intervention" and node2["type"] == "symptom":
                        weight *= 1.4  # Interventions target symptoms
                    
                    # Cap at 1.0
                    weight = min(1.0, weight)
                    
                    self.edges.append({
                        "source": node1["id"],
                        "target": node2["id"],
                        "weight": weight
                    })
        
        # Generate node attributes
        self.node_attributes = {}
        
        for node in self.nodes:
            # Generate demo attribute values
            if node["type"] == "symptom":
                severity = np.round(np.random.uniform(0, 10), 1)
                frequency = np.random.choice(["daily", "weekly", "occasional"])
                self.node_attributes[node["id"]] = {
                    "severity": severity,
                    "frequency": frequency
                }
            elif node["type"] == "factor":
                impact = np.round(np.random.uniform(0, 1), 2)
                modifiable = np.random.choice([True, False])
                self.node_attributes[node["id"]] = {
                    "impact": impact,
                    "modifiable": modifiable
                }
            elif node["type"] == "outcome":
                value = np.round(np.random.uniform(0, 100), 0)
                self.node_attributes[node["id"]] = {
                    "value": value
                }
            else:  # intervention
                efficacy = np.round(np.random.uniform(0, 1), 2)
                cost = np.random.choice(["low", "medium", "high"])
                self.node_attributes[node["id"]] = {
                    "efficacy": efficacy,
                    "cost": cost
                }
        
        return self
    
    def compute_network_metrics(self):
        """
        Compute basic network metrics.
        
        Returns:
            dict: Dictionary of network metrics
        """
        if self.nodes is None or self.edges is None:
            raise ValueError("No network loaded. Call load_network() or generate_demo_network() first.")
        
        # Create node-to-index mapping
        node_indices = {node["id"]: i for i, node in enumerate(self.nodes)}
        
        # Compute degree for each node
        degrees = {node["id"]: 0 for node in self.nodes}
        
        for edge in self.edges:
            degrees[edge["source"]] += 1
            degrees[edge["target"]] += 1
        
        # Compute average degree
        avg_degree = sum(degrees.values()) / len(degrees)
        
        # Compute density
        n_nodes = len(self.nodes)
        max_edges = n_nodes * (n_nodes - 1)
        density = len(self.edges) / max_edges if max_edges > 0 else 0
        
        # Compute clustering coefficient (simplified)
        clustering = 0.0
        
        # Find connected components (simplified)
        components = []
        
        # Identify node types and their counts
        node_types = {}
        for node in self.nodes:
            node_type = node.get("type", "unknown")
            if node_type in node_types:
                node_types[node_type] += 1
            else:
                node_types[node_type] = 1
        
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "avg_degree": avg_degree,
            "density": density,
            "node_types": node_types,
            "max_degree": max(degrees.values()),
            "min_degree": min(degrees.values())
        }
    
    def plot_network(self, color_by: str = "type", title: str = "Network Visualization"):
        """
        Create a network visualization using Plotly.
        
        Args:
            color_by (str): Node attribute to use for coloring
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        if self.nodes is None or self.edges is None:
            raise ValueError("No network loaded. Call load_network() or generate_demo_network() first.")
        
        # Create node positions using a force-directed layout simulation
        node_positions = self._simulate_force_directed_layout()
        
        # Create figure
        fig = go.Figure()
        
        # Add edges as lines
        edge_x = []
        edge_y = []
        edge_weights = []
        
        node_positions_dict = {node["id"]: pos for node, pos in zip(self.nodes, node_positions)}
        
        for edge in self.edges:
            x0, y0 = node_positions_dict[edge["source"]]
            x1, y1 = node_positions_dict[edge["target"]]
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Store edge weight for coloring
            weight = edge.get("weight", 0.5)
            edge_weights.extend([weight, weight, None])
        
        # Add edges trace
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(
                width=1,
                color=edge_weights,
                colorscale='Viridis',
                colorbar=dict(
                    title="Edge Weight",
                    thickness=15,
                    xanchor='left',
                    titleside='right'
                ),
                cmin=0,
                cmax=1
            ),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Get node colors based on attribute
        color_values = []
        color_groups = {}
        
        for node in self.nodes:
            if color_by in node:
                value = node[color_by]
                
                # Track unique values for categorical attributes
                if value not in color_groups:
                    color_groups[value] = len(color_groups)
                
                color_values.append(color_groups[value])
            else:
                # Default color if attribute not found
                color_values.append(0)
        
        # Create node traces
        node_x = [pos[0] for pos in node_positions]
        node_y = [pos[1] for pos in node_positions]
        
        # Prepare node hover text
        node_text = []
        
        for i, node in enumerate(self.nodes):
            text = f"ID: {node['id']}<br>Name: {node.get('name', 'N/A')}"
            
            # Add other node attributes
            for key, value in node.items():
                if key not in ["id", "name"]:
                    text += f"<br>{key}: {value}"
            
            # Add attributes from node_attributes if available
            if self.node_attributes and node["id"] in self.node_attributes:
                for key, value in self.node_attributes[node["id"]].items():
                    text += f"<br>{key}: {value}"
            
            node_text.append(text)
        
        # Add nodes trace
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            marker=dict(
                size=10,
                color=color_values,
                colorscale='Rainbow',
                line=dict(width=1, color='black')
            ),
            text=node_text,
            hoverinfo='text',
            showlegend=False
        ))
        
        # Create legend for node types if coloring by type
        if color_by == "type" and color_groups:
            for type_name, color_idx in color_groups.items():
                # Create a dummy trace for each type for the legend
                fig.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=f'hsl({360 * color_idx / len(color_groups)}, 80%, 50%)',
                        line=dict(width=1, color='black')
                    ),
                    name=type_name,
                    showlegend=True
                ))
        
        # Set layout
        fig.update_layout(
            title=title,
            width=self.width,
            height=self.height,
            showlegend=(color_by == "type"),
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        return fig
    
    def plot_network_metrics(self, title: str = "Network Metrics"):
        """
        Create visualizations of network metrics.
        
        Args:
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure with subplots
        """
        if self.nodes is None or self.edges is None:
            raise ValueError("No network loaded. Call load_network() or generate_demo_network() first.")
        
        # Compute metrics
        metrics = self.compute_network_metrics()
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, 
            cols=2,
            specs=[
                [{"type": "domain"}, {"type": "xy"}],
                [{"type": "bar"}, {"type": "pie"}]
            ],
            subplot_titles=["Node Types", "Degree Distribution", "Network Density", "Node Domain"]
        )
        
        # Node type distribution (pie chart)
        labels = list(metrics["node_types"].keys())
        values = list(metrics["node_types"].values())
        
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                textinfo="label+percent"
            ),
            row=1, col=1
        )
        
        # Compute degree distribution
        degrees = {}
        for edge in self.edges:
            if edge["source"] not in degrees:
                degrees[edge["source"]] = 0
            if edge["target"] not in degrees:
                degrees[edge["target"]] = 0
            
            degrees[edge["source"]] += 1
            degrees[edge["target"]] += 1
        
        degree_counts = {}
        for degree in degrees.values():
            if degree not in degree_counts:
                degree_counts[degree] = 0
            degree_counts[degree] += 1
        
        degree_values = list(sorted(degree_counts.keys()))
        counts = [degree_counts[d] for d in degree_values]
        
        # Degree distribution (scatter plot)
        fig.add_trace(
            go.Bar(
                x=degree_values,
                y=counts,
                marker=dict(color="royalblue")
            ),
            row=1, col=2
        )
        
        # Network metrics (bar chart)
        metric_names = ["n_nodes", "n_edges", "avg_degree", "density"]
        metric_values = [metrics[name] if name != "density" else metrics[name] * 100 for name in metric_names]
        metric_labels = ["Number of Nodes", "Number of Edges", "Average Degree", "Density (%)"]
        
        fig.add_trace(
            go.Bar(
                x=metric_labels,
                y=metric_values,
                marker=dict(color="mediumseagreen")
            ),
            row=2, col=1
        )
        
        # Domain distribution if available
        domain_counts = {}
        for node in self.nodes:
            if "domain" in node:
                domain = node["domain"]
                if domain not in domain_counts:
                    domain_counts[domain] = 0
                domain_counts[domain] += 1
        
        if domain_counts:
            domain_labels = list(domain_counts.keys())
            domain_values = list(domain_counts.values())
            
            fig.add_trace(
                go.Pie(
                    labels=domain_labels,
                    values=domain_values,
                    textinfo="label+percent"
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            width=self.width,
            height=self.height,
            showlegend=False
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Degree", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        
        fig.update_xaxes(title_text="Metric", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        
        return fig
    
    def _simulate_force_directed_layout(self, iterations: int = 100):
        """
        Simulate a force-directed layout algorithm to position nodes.
        
        Args:
            iterations (int): Number of simulation iterations
            
        Returns:
            list: List of (x, y) positions for nodes
        """
        n_nodes = len(self.nodes)
        
        # Initialize random positions in a circle
        positions = []
        for i in range(n_nodes):
            angle = 2 * np.pi * i / n_nodes
            x = np.cos(angle)
            y = np.sin(angle)
            positions.append(np.array([x, y]))
        
        # Create adjacency list for faster lookup
        adjacency = {node["id"]: [] for node in self.nodes}
        for edge in self.edges:
            source = edge["source"]
            target = edge["target"]
            weight = edge.get("weight", 0.5)
            adjacency[source].append((target, weight))
            adjacency[target].append((source, weight))
        
        # Node ID to index mapping
        node_indices = {node["id"]: i for i, node in enumerate(self.nodes)}
        
        # Force-directed layout simulation
        for _ in range(iterations):
            # Compute repulsive forces between all nodes
            forces = [np.zeros(2) for _ in range(n_nodes)]
            
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        # Repulsive force
                        diff = positions[i] - positions[j]
                        distance = np.linalg.norm(diff)
                        if distance < 0.01:  # Avoid division by zero
                            diff = np.random.randn(2) * 0.01
                            distance = np.linalg.norm(diff)
                        
                        # Force is inversely proportional to square of distance
                        repulsion = 0.2 / (distance ** 2)
                        forces[i] += repulsion * diff / distance
            
            # Compute attractive forces for connected nodes
            for i, node in enumerate(self.nodes):
                node_id = node["id"]
                for neighbor_id, weight in adjacency[node_id]:
                    j = node_indices[neighbor_id]
                    
                    # Attractive force
                    diff = positions[j] - positions[i]
                    distance = np.linalg.norm(diff)
                    if distance < 0.01:  # Avoid division by zero
                        continue
                    
                    # Force is proportional to distance and edge weight
                    attraction = 0.05 * distance * weight
                    forces[i] += attraction * diff / distance
            
            # Update positions
            for i in range(n_nodes):
                # Limit force magnitude
                force_mag = np.linalg.norm(forces[i])
                if force_mag > 0.2:
                    forces[i] = forces[i] / force_mag * 0.2
                
                positions[i] += forces[i]
        
        # Center positions
        positions = np.array(positions)
        center = np.mean(positions, axis=0)
        positions = positions - center
        
        # Scale positions to fit in [-1, 1] x [-1, 1]
        max_distance = np.max(np.abs(positions))
        if max_distance > 0:
            positions = positions / max_distance
        
        return positions.tolist()


class TimeSeriesVisualizer:
    """Class for time series visualization and analysis."""
    
    def __init__(self, width=800, height=600):
        """
        Initialize the time series visualizer.
        
        Args:
            width (int): Width of the visualization
            height (int): Height of the visualization
        """
        self.width = width
        self.height = height
        self.time_series_data = None
        self.timestamps = None
        self.variables = None
        self.metadata = None
    
    def load_time_series(self, data, timestamps=None, variables=None, metadata=None):
        """
        Load time series data.
        
        Args:
            data (np.ndarray): 2D array of time series data with shape (n_timestamps, n_variables)
            timestamps (list, optional): List of timestamp strings or datetime objects
            variables (list, optional): List of variable names
            metadata (dict, optional): Dictionary of metadata
            
        Returns:
            self: For method chaining
        """
        self.time_series_data = data
        self.timestamps = timestamps
        self.variables = variables
        self.metadata = metadata
        
        return self
    
    def generate_demo_data(self, n_timestamps: int = 100, n_variables: int = 5):
        """
        Generate demo time series data.
        
        Args:
            n_timestamps (int): Number of timestamps
            n_variables (int): Number of variables
            
        Returns:
            self: For method chaining
        """
        # Generate timestamps
        start_date = datetime.datetime(2023, 1, 1)
        self.timestamps = [start_date + datetime.timedelta(days=i) for i in range(n_timestamps)]
        
        # Generate variable names
        var_names = ["Depression", "Anxiety", "Sleep Quality", "Stress Level", "Physical Activity",
                     "Social Interaction", "Medication Adherence", "Mood", "Energy", "Appetite"]
        self.variables = var_names[:n_variables]
        
        # Generate data with trends, seasonality, and correlations
        self.time_series_data = np.zeros((n_timestamps, n_variables))
        
        # Base signals
        t = np.arange(n_timestamps)
        
        # Add trend to first variable (e.g., depression decreasing)
        trend = -0.5 * (t / n_timestamps) + np.random.normal(0, 0.1, n_timestamps)
        self.time_series_data[:, 0] = 0.7 + 0.3 * trend
        
        # Add weekly seasonality to second variable (e.g., anxiety)
        seasonality = 0.2 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 0.1, n_timestamps)
        self.time_series_data[:, 1] = 0.5 + seasonality
        
        # Third variable correlated with first but delayed (e.g., sleep quality improves after depression decreases)
        delay = 5
        correlated = np.zeros(n_timestamps)
        correlated[delay:] = trend[:-delay]
        correlated += np.random.normal(0, 0.15, n_timestamps)
        self.time_series_data[:, 2] = 0.4 + 0.4 * correlated
        
        # Fourth variable with random spikes (e.g., stress)
        random_spikes = np.random.normal(0, 0.1, n_timestamps)
        spike_indices = np.random.choice(n_timestamps, 10, replace=False)
        random_spikes[spike_indices] += np.random.uniform(0.3, 0.7, 10)
        self.time_series_data[:, 3] = 0.3 + random_spikes
        
        # Fifth variable with long-term oscillation (e.g., physical activity)
        long_oscillation = 0.3 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 0.1, n_timestamps)
        self.time_series_data[:, 4] = 0.6 + long_oscillation
        
        # Add any additional variables with combinations of the above patterns
        for i in range(5, n_variables):
            # Mix of existing signals with random weights
            weights = np.random.uniform(0, 1, 5)
            weights /= weights.sum()  # Normalize
            
            mixed_signal = np.zeros(n_timestamps)
            for j in range(5):
                mixed_signal += weights[j] * self.time_series_data[:, j]
            
            # Add noise
            mixed_signal += np.random.normal(0, 0.1, n_timestamps)
            
            # Scale to [0, 1]
            mixed_signal = (mixed_signal - mixed_signal.min()) / (mixed_signal.max() - mixed_signal.min())
            
            self.time_series_data[:, i] = 0.2 + 0.6 * mixed_signal
        
        # Add metadata
        self.metadata = {
            "subject_id": "DEMO-001",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": self.timestamps[-1].strftime("%Y-%m-%d"),
            "sampling_frequency": "daily",
            "description": "Demo time series data for mental health variables"
        }
        
        return self
    
    def plot_time_series(self, variables=None, normalize=False, title: str = "Time Series Visualization"):
        """
        Create a time series plot.
        
        Args:
            variables (list, optional): List of variable indices or names to plot. If None, all variables are plotted.
            normalize (bool): Whether to normalize each variable to [0, 1]
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        if self.time_series_data is None:
            raise ValueError("No time series data loaded. Call load_time_series() or generate_demo_data() first.")
        
        # Convert variable names to indices if needed
        var_indices = []
        if variables is not None:
            if isinstance(variables[0], str) and self.variables is not None:
                var_dict = {name: i for i, name in enumerate(self.variables)}
                var_indices = [var_dict[var] for var in variables if var in var_dict]
            else:
                var_indices = variables
        else:
            var_indices = list(range(self.time_series_data.shape[1]))
        
        # Create figure
        fig = go.Figure()
        
        # Create x-axis values
        if self.timestamps is not None:
            x_values = self.timestamps
        else:
            x_values = list(range(self.time_series_data.shape[0]))
        
        # Add traces for each variable
        for i in var_indices:
            y_values = self.time_series_data[:, i]
            
            if normalize:
                # Normalize to [0, 1]
                y_min = np.min(y_values)
                y_max = np.max(y_values)
                if y_max > y_min:
                    y_values = (y_values - y_min) / (y_max - y_min)
            
            # Variable name for legend
            var_name = self.variables[i] if self.variables is not None else f"Variable {i}"
            
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines+markers',
                name=var_name,
                line=dict(width=2),
                marker=dict(size=5)
            ))
        
        # Set layout
        fig.update_layout(
            title=title,
            width=self.width,
            height=self.height,
            xaxis_title="Time",
            yaxis_title="Value" if not normalize else "Normalized Value",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        
        return fig
    
    def plot_correlation_matrix(self, title: str = "Variable Correlation Matrix"):
        """
        Create a correlation matrix visualization.
        
        Args:
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        if self.time_series_data is None:
            raise ValueError("No time series data loaded. Call load_time_series() or generate_demo_data() first.")
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(self.time_series_data.T)
        
        # Create variable labels
        if self.variables is not None:
            labels = self.variables
        else:
            labels = [f"Var {i}" for i in range(corr_matrix.shape[0])]
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            x=labels,
            y=labels,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title=title
        )
        
        # Add correlation values as text
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"{corr_matrix[i, j]:.2f}",
                    showarrow=False,
                    font=dict(
                        color="white" if abs(corr_matrix[i, j]) > 0.5 else "black"
                    )
                )
        
        # Set layout
        fig.update_layout(
            width=self.width,
            height=self.height
        )
        
        return fig
    
    def decompose_time_series(self, variable_index=0):
        """
        Decompose a time series into trend, seasonality, and residual components.
        
        Args:
            variable_index (int): Index of the variable to decompose
            
        Returns:
            dict: Dictionary containing the decomposition results
        """
        if self.time_series_data is None:
            raise ValueError("No time series data loaded. Call load_time_series() or generate_demo_data() first.")
        
        # Extract the time series
        time_series = self.time_series_data[:, variable_index]
        
        # Simple moving average for trend estimation
        window_size = min(21, len(time_series) // 3)
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd window size
        
        # Compute trend using moving average
        trend = np.convolve(time_series, np.ones(window_size) / window_size, mode='valid')
        
        # Pad trend to match original length
        pad_size = (len(time_series) - len(trend)) // 2
        trend = np.pad(trend, (pad_size, len(time_series) - len(trend) - pad_size), mode='edge')
        
        # Detrended series
        detrended = time_series - trend
        
        # Estimate seasonality using autocorrelation
        acf = self._autocorrelation(detrended)
        
        # Find peaks in autocorrelation
        peaks = []
        for i in range(1, len(acf) - 1):
            if acf[i] > acf[i-1] and acf[i] > acf[i+1] and acf[i] > 0.1:
                peaks.append(i)
        
        # Estimate season length as the first peak or default to 7 (weekly)
        season_length = peaks[0] if peaks else 7
        
        # Compute seasonal component
        seasonal = np.zeros_like(time_series)
        
        if len(time_series) >= 2 * season_length:
            # Estimate seasonal pattern by averaging
            pattern = np.zeros(season_length)
            
            for i in range(season_length):
                values = time_series[i::season_length]
                pattern[i] = np.mean(values)
            
            # Normalize pattern to have zero mean
            pattern = pattern - np.mean(pattern)
            
            # Replicate pattern
            for i in range(len(time_series)):
                seasonal[i] = pattern[i % season_length]
        
        # Compute residual
        residual = time_series - trend - seasonal
        
        # Prepare result
        var_name = self.variables[variable_index] if self.variables is not None else f"Variable {variable_index}"
        
        return {
            "variable": var_name,
            "original": time_series,
            "trend": trend,
            "seasonal": seasonal,
            "residual": residual,
            "season_length": season_length
        }
    
    def plot_decomposition(self, variable_index=0, title: str = "Time Series Decomposition"):
        """
        Create a visualization of time series decomposition.
        
        Args:
            variable_index (int): Index of the variable to decompose
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure
        """
        # Decompose the time series
        decomp = self.decompose_time_series(variable_index)
        
        # Create x-axis values
        if self.timestamps is not None:
            x_values = self.timestamps
        else:
            x_values = list(range(len(decomp["original"])))
        
        # Create figure with subplots
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            subplot_titles=["Original", "Trend", "Seasonal", "Residual"],
            vertical_spacing=0.1
        )
        
        # Add traces
        components = ["original", "trend", "seasonal", "residual"]
        colors = ["blue", "green", "red", "purple"]
        
        for i, (component, color) in enumerate(zip(components, colors)):
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=decomp[component],
                    mode='lines',
                    name=component.capitalize(),
                    line=dict(color=color)
                ),
                row=i+1,
                col=1
            )
        
        # Set layout
        var_name = decomp["variable"]
        
        fig.update_layout(
            title=f"{title}: {var_name}",
            width=self.width,
            height=self.height,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Trend", row=2, col=1)
        fig.update_yaxes(title_text="Seasonal", row=3, col=1)
        fig.update_yaxes(title_text="Residual", row=4, col=1)
        
        # Update x-axis title
        fig.update_xaxes(title_text="Time", row=4, col=1)
        
        return fig
    
    def _autocorrelation(self, x, max_lag=None):
        """
        Compute autocorrelation function.
        
        Args:
            x (np.ndarray): 1D array
            max_lag (int, optional): Maximum lag. If None, defaults to len(x)//2.
            
        Returns:
            np.ndarray: Autocorrelation function
        """
        if max_lag is None:
            max_lag = len(x) // 2
        
        # Mean and variance
        mean = np.mean(x)
        var = np.var(x)
        
        # Normalize
        x = x - mean
        
        # Compute autocorrelation
        acf = np.zeros(max_lag + 1)
        
        for lag in range(max_lag + 1):
            # Cross-correlation at each lag
            c = np.sum(x[lag:] * x[:(len(x)-lag)])
            acf[lag] = c / ((len(x) - lag) * var)
        
        return acf


def create_advanced_visualization_page(st):
    """
    Create a Streamlit interface for advanced visualizations.
    
    Args:
        st: Streamlit module
    """
    st.title("Advanced Data Visualization")
    
    # Create tabs for different visualization types
    viz_tabs = st.tabs([
        "Brain Imaging", 
        "Network Analysis", 
        "Time Series Analysis", 
        "Multimodal Integration"
    ])
    
    # Brain Imaging Visualization
    with viz_tabs[0]:
        st.header("3D Brain Visualization")
        
        st.markdown("""
        This module enables interactive 3D visualization of brain imaging data,
        including region activation, connectivity, and comparison across conditions.
        """)
        
        # Brain visualization options
        viz_type = st.radio(
            "Visualization Type",
            ["Brain Regions", "Connectome", "Region Comparison"],
            horizontal=True
        )
        
        # Demo data generation
        if st.button("Generate Demo Brain Data"):
            # Create brain visualizer with demo data
            brain_viz = Brain3DVisualizer(width=800, height=600)
            brain_viz.generate_demo_data(n_regions=100)
            
            # Store in session state
            st.session_state.brain_viz = brain_viz
            st.success("Demo brain data generated successfully!")
        
        # Display visualization if data exists
        if "brain_viz" in st.session_state:
            brain_viz = st.session_state.brain_viz
            
            if viz_type == "Brain Regions":
                # Create and display brain regions plot
                fig = brain_viz.plot_3d_brain(title="3D Brain Region Activation")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                This visualization shows activation patterns across different brain regions.
                Each point represents a brain region, with color and size indicating activation level.
                Hover over points to see region details.
                """)
            
            elif viz_type == "Connectome":
                # Connectivity options
                density = st.slider("Connection Density", 0.01, 0.5, 0.1, 0.01)
                threshold = st.slider("Connection Threshold", 0.0, 0.5, 0.1, 0.01)
                
                # Create connectivity matrix
                connectivity = brain_viz.create_connectome_data(density=density)
                
                # Create and display connectome plot
                fig = brain_viz.plot_connectome(connectivity=connectivity, threshold=threshold, title="Brain Connectome")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                The connectome visualization shows connectivity between brain regions.
                Lines represent connections, with color indicating connection strength.
                Adjust the density and threshold parameters to control the visualization complexity.
                """)
            
            elif viz_type == "Region Comparison":
                # Create and display region comparison plot
                fig = brain_viz.plot_region_comparison(title="Brain Region Activation Comparison")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                This bar chart compares activation levels across different brain regions.
                It helps identify which regions show the highest activation or deactivation.
                """)
    
    # Network Analysis Visualization
    with viz_tabs[1]:
        st.header("Network Analysis")
        
        st.markdown("""
        This module provides tools for visualizing and analyzing networks and relationships
        between symptoms, factors, and outcomes in mental health data.
        """)
        
        # Network visualization options
        network_viz_type = st.radio(
            "Network Visualization Type",
            ["Network Graph", "Network Metrics"],
            horizontal=True
        )
        
        # Color by options
        color_by = st.selectbox(
            "Color Nodes By",
            ["type", "domain"]
        )
        
        # Demo data generation
        if st.button("Generate Demo Network"):
            # Create network visualizer with demo data
            net_viz = NetworkVisualizer(width=800, height=600)
            net_viz.generate_demo_network(n_nodes=50, connection_density=0.1)
            
            # Store in session state
            st.session_state.net_viz = net_viz
            st.success("Demo network generated successfully!")
        
        # Display visualization if data exists
        if "net_viz" in st.session_state:
            net_viz = st.session_state.net_viz
            
            if network_viz_type == "Network Graph":
                # Create and display network graph
                fig = net_viz.plot_network(color_by=color_by, title="Mental Health Factor Network")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                This network visualization shows relationships between different mental health factors.
                Nodes represent factors, symptoms, or outcomes, and edges represent connections between them.
                The color of each node indicates its type or domain, and the width of edges represents connection strength.
                Hover over nodes for additional information.
                """)
            
            elif network_viz_type == "Network Metrics":
                # Calculate and display network metrics
                metrics = net_viz.compute_network_metrics()
                
                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Nodes", metrics["n_nodes"])
                
                with col2:
                    st.metric("Edges", metrics["n_edges"])
                
                with col3:
                    st.metric("Avg. Degree", f"{metrics['avg_degree']:.2f}")
                
                with col4:
                    st.metric("Density", f"{metrics['density']:.3f}")
                
                # Create and display network metrics visualization
                fig = net_viz.plot_network_metrics(title="Network Metrics Overview")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                This dashboard shows key metrics of the mental health factor network.
                It provides insights into the structure and complexity of relationships between factors.
                The visualizations include node type distribution, degree distribution, and overall network statistics.
                """)
    
    # Time Series Analysis Visualization
    with viz_tabs[2]:
        st.header("Time Series Analysis")
        
        st.markdown("""
        This module provides tools for visualizing and analyzing longitudinal mental health data,
        including trends, seasonality, and correlations between different variables.
        """)
        
        # Time series visualization options
        ts_viz_type = st.radio(
            "Time Series Visualization Type",
            ["Multivariate Plot", "Correlation Matrix", "Decomposition"],
            horizontal=True
        )
        
        # Demo data generation
        if st.button("Generate Demo Time Series"):
            # Create time series visualizer with demo data
            ts_viz = TimeSeriesVisualizer(width=800, height=500)
            ts_viz.generate_demo_data(n_timestamps=100, n_variables=5)
            
            # Store in session state
            st.session_state.ts_viz = ts_viz
            st.success("Demo time series generated successfully!")
        
        # Display visualization if data exists
        if "ts_viz" in st.session_state:
            ts_viz = st.session_state.ts_viz
            
            if ts_viz_type == "Multivariate Plot":
                # Variable selection
                if ts_viz.variables:
                    selected_vars = st.multiselect(
                        "Select Variables to Display",
                        ts_viz.variables,
                        default=ts_viz.variables[:3]
                    )
                    
                    # Normalize option
                    normalize = st.checkbox("Normalize Variables", value=True)
                    
                    if selected_vars:
                        # Create and display time series plot
                        fig = ts_viz.plot_time_series(variables=selected_vars, normalize=normalize, 
                                                     title="Mental Health Variables Over Time")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Please select at least one variable to display.")
                else:
                    # Create and display time series plot with all variables
                    fig = ts_viz.plot_time_series(normalize=True, title="Mental Health Variables Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                This time series visualization shows how different mental health variables change over time.
                It can help identify trends, patterns, and potential relationships between variables.
                Use the normalize option to compare variables on the same scale.
                """)
            
            elif ts_viz_type == "Correlation Matrix":
                # Create and display correlation matrix
                fig = ts_viz.plot_correlation_matrix(title="Variable Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                The correlation matrix shows how different mental health variables are related to each other.
                Strong positive correlations (close to 1) appear in red, while strong negative correlations (close to -1) 
                appear in blue. Values close to 0 indicate little or no correlation.
                """)
            
            elif ts_viz_type == "Decomposition":
                # Variable selection for decomposition
                if ts_viz.variables:
                    selected_var = st.selectbox(
                        "Select Variable to Decompose",
                        ts_viz.variables,
                        index=0
                    )
                    
                    var_idx = ts_viz.variables.index(selected_var)
                else:
                    var_idx = 0
                
                # Create and display decomposition plot
                fig = ts_viz.plot_decomposition(variable_index=var_idx, title="Time Series Decomposition")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                Time series decomposition breaks down a variable into its component parts:
                - Trend: The long-term progression of the series
                - Seasonal: Regular pattern of variation
                - Residual: The irregular component (what remains after removing trend and seasonality)
                
                This analysis helps understand the underlying patterns in mental health data.
                """)
    
    # Multimodal Integration Visualization
    with viz_tabs[3]:
        st.header("Multimodal Data Integration")
        
        st.markdown("""
        This module visualizes how different data modalities (text, audio, physiological, imaging)
        are integrated and contribute to mental health assessments.
        """)
        
        # Demo visualization options
        integration_viz_type = st.radio(
            "Integration Visualization Type",
            ["Modality Contribution", "Feature Importance", "Multimodal Embeddings"],
            horizontal=True
        )
        
        # Display based on selected visualization type
        if integration_viz_type == "Modality Contribution":
            # Create pie chart of modality contributions
            modalities = ["Text", "Audio", "Physiological", "Imaging"]
            contributions = [0.45, 0.35, 0.15, 0.05]
            
            fig = px.pie(
                values=contributions,
                names=modalities,
                title="Modality Contribution to Overall Assessment",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4
            )
            
            fig.update_layout(
                annotations=[dict(text="Contribution", x=0.5, y=0.5, font_size=15, showarrow=False)]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display condition-specific contributions
            st.subheader("Condition-Specific Modality Contributions")
            
            conditions = ["Depression", "Anxiety", "PTSD", "Bipolar", "Schizophrenia"]
            selected_condition = st.selectbox("Select Condition", conditions)
            
            # Different contributions for each condition
            condition_contributions = {
                "Depression": [0.55, 0.30, 0.10, 0.05],
                "Anxiety": [0.40, 0.45, 0.10, 0.05],
                "PTSD": [0.35, 0.35, 0.20, 0.10],
                "Bipolar": [0.40, 0.30, 0.20, 0.10],
                "Schizophrenia": [0.35, 0.25, 0.25, 0.15]
            }
            
            fig = px.pie(
                values=condition_contributions[selected_condition],
                names=modalities,
                title=f"Modality Contribution for {selected_condition}",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            These charts show how different data modalities contribute to mental health assessments.
            The contribution varies by condition, reflecting the different diagnostic signals captured by each modality.
            """)
        
        elif integration_viz_type == "Feature Importance":
            # Create bar chart of feature importance across modalities
            modalities = ["Text", "Audio", "Physiological", "Imaging"]
            
            # Generate feature names and importance scores for each modality
            features_data = []
            
            # Text features
            text_features = ["Negative words", "Emotional content", "Topic coherence", 
                            "Self-references", "Anxiety indicators"]
            text_importance = [0.85, 0.72, 0.65, 0.58, 0.45]
            
            # Audio features
            audio_features = ["Voice pitch", "Speech rate", "Vocal energy", 
                             "Pause frequency", "Vocal quality"]
            audio_importance = [0.78, 0.65, 0.62, 0.55, 0.48]
            
            # Physiological features
            physio_features = ["Heart rate var.", "Skin conductance", "Respiration", 
                              "Movement patterns", "Sleep quality"]
            physio_importance = [0.72, 0.68, 0.60, 0.52, 0.45]
            
            # Imaging features
            imaging_features = ["Amygdala activation", "Prefrontal activity", "Hippocampal volume", 
                               "Cingulate response", "Network connectivity"]
            imaging_importance = [0.80, 0.75, 0.68, 0.62, 0.55]
            
            # Combine all features
            for i, (feat, imp) in enumerate(zip(text_features, text_importance)):
                features_data.append({"Feature": feat, "Importance": imp, "Modality": "Text", "Rank": i+1})
            
            for i, (feat, imp) in enumerate(zip(audio_features, audio_importance)):
                features_data.append({"Feature": feat, "Importance": imp, "Modality": "Audio", "Rank": i+1})
            
            for i, (feat, imp) in enumerate(zip(physio_features, physio_importance)):
                features_data.append({"Feature": feat, "Importance": imp, "Modality": "Physiological", "Rank": i+1})
            
            for i, (feat, imp) in enumerate(zip(imaging_features, imaging_importance)):
                features_data.append({"Feature": feat, "Importance": imp, "Modality": "Imaging", "Rank": i+1})
            
            # Convert to DataFrame
            df = pd.DataFrame(features_data)
            
            # Create bar chart
            fig = px.bar(
                df,
                x="Importance",
                y="Feature",
                color="Modality",
                facet_col="Modality",
                facet_col_wrap=2,
                height=700,
                title="Feature Importance by Modality",
                color_discrete_sequence=px.colors.qualitative.Set3,
                labels={"Importance": "Relative Importance"}
            )
            
            # Update layout
            fig.update_yaxes(matches=None)
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            This visualization shows the most important features from each data modality.
            It highlights what signals the model uses from each modality to make its assessments.
            """)
        
        elif integration_viz_type == "Multimodal Embeddings":
            st.markdown("### Multimodal Embedding Visualization")
            
            st.markdown("""
            This visualization shows how samples from different data modalities are embedded in a shared 
            latent space after multimodal integration.
            """)
            
            # Generate demo embedding data
            n_samples = 100
            n_conditions = 3
            
            # Create embedding coordinates with clusters
            np.random.seed(42)
            
            # Generate cluster centers
            centers = np.array([
                [1, 1],     # Depression
                [-1, -1],   # Anxiety
                [0, -1.5]   # PTSD
            ])
            
            # Generate cluster points
            embeddings = []
            condition_labels = []
            modality_labels = []
            
            for i in range(n_conditions):
                n_cluster = n_samples // n_conditions
                
                # Different spread for each modality to show modality-specific patterns
                modalities = ["Text", "Audio", "Physiological", "Imaging"]
                spreads = [0.2, 0.3, 0.4, 0.5]
                
                for modality, spread in zip(modalities, spreads):
                    # Generate points around cluster center
                    points = centers[i] + np.random.randn(n_cluster // 4, 2) * spread
                    
                    embeddings.extend(points)
                    
                    if i == 0:
                        condition_labels.extend(["Depression"] * (n_cluster // 4))
                    elif i == 1:
                        condition_labels.extend(["Anxiety"] * (n_cluster // 4))
                    else:
                        condition_labels.extend(["PTSD"] * (n_cluster // 4))
                    
                    modality_labels.extend([modality] * (n_cluster // 4))
            
            # Convert to DataFrame
            embedding_df = pd.DataFrame({
                "x": [p[0] for p in embeddings],
                "y": [p[1] for p in embeddings],
                "Condition": condition_labels,
                "Modality": modality_labels
            })
            
            # Create visualization options
            color_by = st.radio(
                "Color By",
                ["Condition", "Modality"],
                horizontal=True
            )
            
            # Create scatter plot
            fig = px.scatter(
                embedding_df,
                x="x",
                y="y",
                color=color_by,
                symbol="Modality" if color_by == "Condition" else None,
                title="Multimodal Embedding Space",
                height=600,
                labels={"x": "Dimension 1", "y": "Dimension 2"}
            )
            
            # Update layout
            fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            This embedding visualization shows how multimodal data is integrated into a shared 
            representation space. Points that are closer together are more similar in their multimodal patterns.
            
            You can color the points by condition to see how well the conditions separate in the embedding space,
            or by modality to see if there are modality-specific patterns.
            """)