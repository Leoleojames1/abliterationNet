"use client";

import { useState, useEffect } from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Loader2 } from 'lucide-react';
import DatasetUploader from './DatasetUploader';
import { NeuralNetworkVisualizer } from './NeuralNetworkVisualizer';

const AbliteratorController = () => {
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [initialized, setInitialized] = useState(false);
  const [visualData, setVisualData] = useState({
    layers: [
      [{ activation: 0 }, { activation: 0 }, { activation: 0 }],
      [{ activation: 0 }, { activation: 0 }, { activation: 0 }, { activation: 0 }],
      [{ activation: 0 }, { activation: 0 }]
    ],
    synapses: [
      { start: [-1, 0, 0], end: [1, 0, 0], weight: 0.5 },
      { start: [-1, 0.5, 0], end: [1, 0.5, 0], weight: -0.3 },
      { start: [1, 0, 0], end: [3, 0, 0], weight: 0.8 }
    ]
  });

  const transformAbliteratorData = (data) => {
    if (!data.modified_layers || Object.keys(data.modified_layers).length === 0) {
      return visualData; // Return current state if no new data
    }

    const layers = [];
    const synapses = [];
    const layerSpacing = 2;
    
    Object.entries(data.modified_layers).forEach(([layerType, layerData], layerIndex) => {
      const xPos = layerIndex * layerSpacing - Object.keys(data.modified_layers).length;
      
      // Create neurons with activation values
      const neurons = Array(data.hidden_size || 768).fill(0).map((_, i) => ({
        activation: data.activation_values?.[`${layerType}.${layerIndex}`]?.[i] || Math.random(),
        id: `${layerType}-${layerIndex}-${i}`
      }));
      
      layers.push(neurons);
      
      // Create synapses between consecutive layers
      if (layerIndex > 0 && layers[layerIndex - 1]) {
        const prevLayer = layers[layerIndex - 1];
        neurons.forEach((neuron, targetIdx) => {
          prevLayer.forEach((prevNeuron, sourceIdx) => {
            const weight = layerData?.[targetIdx]?.[sourceIdx] || Math.random() * 2 - 1;
            synapses.push({
              start: [
                (layerIndex - 1) * layerSpacing - Object.keys(data.modified_layers).length,
                sourceIdx * 0.5 - (prevLayer.length - 1) * 0.25,
                0
              ],
              end: [
                xPos,
                targetIdx * 0.5 - (neurons.length - 1) * 0.25,
                0
              ],
              weight: weight
            });
          });
        });
      }
    });

    return { layers, synapses };
  };

  const handleDatasetLoaded = async (data) => {
    setError(null);
    setLoading(true);
    
    try {
      // Initialize abliterator if not already initialized
      if (!initialized) {
        await initializeAbliterator();
      }
      
      // Fetch new network state after dataset is loaded
      await fetchNetworkState();
      setInitialized(true);
    } catch (err) {
      setError('Failed to process dataset: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const initializeAbliterator = async () => {
    const response = await fetch('http://localhost:8000/initialize_reverse_abliterator', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_path: '/models/mistral-7b',  // Update with your model path
        dataset: [[], []], // Empty dataset - will be populated by uploader
        device: "cuda",
        activation_layers: ['resid_pre', 'resid_post', 'mlp_out', 'attn_out']
      })
    });
    
    if (!response.ok) throw new Error('Failed to initialize abliterator');
  };

  const fetchNetworkState = async () => {
    try {
      const response = await fetch('http://localhost:8000/network_state');
      const data = await response.json();
      const transformedData = transformAbliteratorData(data);
      setVisualData(transformedData);
    } catch (err) {
      setError('Failed to fetch network state: ' + err.message);
    }
  };

  const enhanceModel = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/enhance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strength: 0.1,
          W_O: true,
          mlp: true
        })
      });

      if (!response.ok) throw new Error('Enhancement failed');
      await fetchNetworkState();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (initialized) {
      const interval = setInterval(fetchNetworkState, 5000);
      return () => clearInterval(interval);
    }
  }, [initialized]);

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto p-4">
        <Card className="mb-4">
          <CardHeader>
            <CardTitle>Abliterator Neural Network Visualization</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col lg:flex-row gap-4">
              <div className="w-full lg:w-1/3">
                <DatasetUploader onDatasetLoaded={handleDatasetLoaded} />
                
                <div className="flex gap-2 mt-4">
                  <Button 
                    onClick={enhanceModel} 
                    disabled={loading || !initialized}
                    className="w-full"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Processing
                      </>
                    ) : (
                      'Enhance Model'
                    )}
                  </Button>
                  <Button 
                    onClick={fetchNetworkState} 
                    disabled={loading || !initialized}
                    variant="outline"
                    className="w-full"
                  >
                    Refresh
                  </Button>
                </div>
                
                {error && (
                  <Alert variant="destructive" className="mt-4">
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                )}
              </div>
              
              <div className="w-full lg:w-2/3 h-[600px] bg-gray-900 rounded-lg">
                <NeuralNetworkVisualizer initialData={visualData} />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default AbliteratorController;