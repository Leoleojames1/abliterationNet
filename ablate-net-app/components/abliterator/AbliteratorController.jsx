"use client";

import { useState, useEffect } from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Loader2 } from 'lucide-react';
import DatasetUploader from './DatasetUploader';
import { NeuralNetworkVisualizer } from './NeuralNetworkVisualizer';
import ModelManager from './ModelManager';
import ProfileSettings from './ProfileSettings';

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
  const [mode, setMode] = useState('demo');

  const handleDatasetLoaded = async (data) => {
    setError(null);
    setLoading(true);
    
    try {
      if (!initialized) {
        await initializeAbliterator();
      }
      
      await fetchNetworkState();
      setInitialized(true);
    } catch (err) {
      setError('Failed to process dataset: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const initializeAbliterator = async (modelPath = '/models/mistral-7b') => {
    const response = await fetch('http://localhost:8000/initialize_reverse_abliterator', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_path: modelPath,
        dataset: [[], []], // Empty dataset - will be populated by uploader
        device: "cuda",
        activation_layers: ['resid_pre', 'resid_post', 'mlp_out', 'attn_out']
      })
    });
    
    if (!response.ok) throw new Error('Failed to initialize abliterator');
  };

  const fetchNetworkState = async () => {
    try {
      const response = await fetch('http://localhost:8000/test_reverse_abliterator');
      if (!response.ok) throw new Error('Failed to fetch network state');
      const data = await response.json();
      setVisualData(transformAbliteratorData(data.results));
    } catch (err) {
      setError('Failed to fetch network state: ' + err.message);
    }
  };

  const transformAbliteratorData = (data) => {
    // This function would need to be implemented to transform the API response
    // into the format expected by NeuralNetworkVisualizer
    // For now, we'll return a placeholder with random data
    return {
      layers: [
        [{ activation: Math.random() }, { activation: Math.random() }, { activation: Math.random() }],
        [{ activation: Math.random() }, { activation: Math.random() }, { activation: Math.random() }, { activation: Math.random() }],
        [{ activation: Math.random() }, { activation: Math.random() }]
      ],
      synapses: [
        { start: [-1, 0, 0], end: [1, 0, 0], weight: Math.random() * 2 - 1 },
        { start: [-1, 0.5, 0], end: [1, 0.5, 0], weight: Math.random() * 2 - 1 },
        { start: [1, 0, 0], end: [3, 0, 0], weight: Math.random() * 2 - 1 }
      ]
    };
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

  const handleModelSelected = async (modelPath) => {
    setError(null);
    setLoading(true);
    
    try {
      await initializeAbliterator(modelPath);
      setInitialized(true);
      await fetchNetworkState();
    } catch (err) {
      setError('Failed to initialize abliterator: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const pushToHuggingFace = () => {
    // This function will be implemented later
    console.log('Pushing model to Hugging Face...');
  };

  useEffect(() => {
    if (mode === 'demo') {
      const interval = setInterval(() => {
        setVisualData(prevData => transformAbliteratorData(prevData));
      }, 1000);
      return () => clearInterval(interval);
    } else if (initialized) {
      const interval = setInterval(fetchNetworkState, 5000);
      return () => clearInterval(interval);
    }
  }, [mode, initialized]);

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto p-4">
        <Card className="mb-4">
          <CardHeader>
            <CardTitle>Abliterator Neural Network Visualization</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs value={mode} onValueChange={setMode} className="mb-4">
              <TabsList>
                <TabsTrigger value="demo">Demo Mode</TabsTrigger>
                <TabsTrigger value="run">Run Mode</TabsTrigger>
              </TabsList>
            </Tabs>

            <div className="flex flex-col lg:flex-row gap-4">
              <div className="w-full lg:w-1/3">
                {mode === 'run' && (
                  <>
                    <ModelManager onModelSelected={handleModelSelected} />
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
                        onClick={pushToHuggingFace} 
                        disabled={!initialized}
                        variant="outline"
                        className="w-full"
                      >
                        Push to HuggingFace
                      </Button>
                    </div>
                  </>
                )}
                
                <ProfileSettings />
                
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

