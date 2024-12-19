import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Database, Upload, Download, Save } from 'lucide-react';

const ModelManager = ({ onModelSelected }) => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [hfModel, setHfModel] = useState('');
  const [saveName, setSaveName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleFileChange = async (event) => {
    const files = Array.from(event.target.files);
    
    // Filter for safetensor files
    const safetensorFiles = files.filter(file => 
      file.name.endsWith('.safetensors') && !file.name.endsWith('.index.json')
    );

    if (safetensorFiles.length > 0) {
      setSelectedFiles(safetensorFiles);
      
      // Get the directory path from the first file
      const dirPath = safetensorFiles[0].webkitRelativePath.split('/')[0];
      const formData = new FormData();
      
      // Add all safetensor files to FormData
      safetensorFiles.forEach(file => {
        formData.append('model_files', file, file.name);
      });

      // Add the model path and other config
      formData.append('config', JSON.stringify({
        model_path: dirPath,
        device: "cuda",
        activation_layers: ['resid_pre', 'resid_post', 'mlp_out', 'attn_out']
      }));
      
      try {
        setLoading(true);
        const response = await fetch('http://localhost:8000/upload_model', {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          const error = await response.json();
          throw new Error(error.detail || 'Failed to load model');
        }

        const data = await response.json();
        
        // Now initialize the abliterator with the uploaded model path
        const initResponse = await fetch('http://localhost:8000/initialize_abliterator', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            model_path: data.model_path,
            dataset: [[], []], // Empty dataset, will be populated later
            device: "cuda",
            activation_layers: ['resid_pre', 'resid_post', 'mlp_out', 'attn_out']
          })
        });

        if (!initResponse.ok) {
          const error = await initResponse.json();
          throw new Error(error.detail || 'Failed to initialize model');
        }

        setSuccess(`Model loaded successfully (${safetensorFiles.length} safetensor files)`);
        onModelSelected(data.model_path);
      } catch (error) {
        setError(error.message);
      } finally {
        setLoading(false);
      }
    } else {
      setError('No safetensor files found in selected directory');
    }
  };

  const loadHuggingFaceModel = async () => {
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const response = await fetch('http://localhost:8000/initialize_abliterator', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_path: hfModel,
          dataset: [[], []], // Empty dataset, will be populated later
          device: "cuda",
          activation_layers: ['resid_pre', 'resid_post', 'mlp_out', 'attn_out']
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to load model');
      }

      setSuccess('Model loaded successfully from HuggingFace');
      onModelSelected(hfModel);
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const saveAbliteratedModel = async () => {
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const response = await fetch('http://localhost:8000/save_abliterated_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ save_name: saveName || undefined })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to save model');
      }

      const data = await response.json();
      setSuccess(`Model saved successfully to ${data.path}`);
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="w-full mb-4">
      <CardHeader>
        <CardTitle>Model Manager</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="local">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="local">Local Model</TabsTrigger>
            <TabsTrigger value="huggingface">Load from HF</TabsTrigger>
            <TabsTrigger value="save">Save Model</TabsTrigger>
          </TabsList>

          <TabsContent value="local">
            <div className="space-y-4">
              <div className="flex flex-col gap-2">
                <p className="text-sm text-muted-foreground">
                  Select the directory containing your model's safetensor files
                </p>
                <Input
                  type="file"
                  webkitdirectory=""
                  directory=""
                  multiple
                  onChange={handleFileChange}
                  disabled={loading}
                  className="cursor-pointer"
                />
              </div>
              {selectedFiles.length > 0 && (
                <div className="text-sm text-muted-foreground">
                  <p className="font-medium">Model files detected:</p>
                  <ul className="list-disc pl-4 mt-2">
                    {selectedFiles.map((file, index) => (
                      <li key={index}>{file.name}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </TabsContent>

          <TabsContent value="huggingface">
            <div className="space-y-4">
              <div className="flex gap-2">
                <Input
                  placeholder="e.g. facebook/opt-350m"
                  value={hfModel}
                  onChange={(e) => setHfModel(e.target.value)}
                />
                <Button 
                  onClick={loadHuggingFaceModel}
                  disabled={loading || !hfModel}
                >
                  {loading ? (
                    <div className="h-4 w-4 border-2 border-t-white rounded-full animate-spin" />
                  ) : (
                    <Download className="h-4 w-4" />
                  )}
                  <span className="ml-2">Load</span>
                </Button>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="save">
            <div className="space-y-4">
              <div className="flex gap-2">
                <Input
                  placeholder="Enter save name (optional)"
                  value={saveName}
                  onChange={(e) => setSaveName(e.target.value)}
                />
                <Button 
                  onClick={saveAbliteratedModel}
                  disabled={loading}
                >
                  {loading ? (
                    <div className="h-4 w-4 border-2 border-t-white rounded-full animate-spin" />
                  ) : (
                    <Save className="h-4 w-4" />
                  )}
                  <span className="ml-2">Save</span>
                </Button>
              </div>
              <p className="text-sm text-muted-foreground">
                Save your abliterated model with an optional custom name
              </p>
            </div>
          </TabsContent>
        </Tabs>

        {error && (
          <Alert variant="destructive" className="mt-4">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        
        {success && (
          <Alert variant="default" className="mt-4">
            <AlertDescription>{success}</AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default ModelManager;