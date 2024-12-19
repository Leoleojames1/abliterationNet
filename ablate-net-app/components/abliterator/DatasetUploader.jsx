import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Database } from 'lucide-react';
import { Dropzone, FileList } from './Dropzone';

const DatasetUploader = ({ onDatasetLoaded }) => {
  const [uploadStatus, setUploadStatus] = useState({ type: null, message: '' });
  const [files, setFiles] = useState([]);
  const [hfRepo, setHfRepo] = useState('');
  const [loading, setLoading] = useState(false);

  const processFiles = async (acceptedFiles) => {
    const newFiles = acceptedFiles.map(file => ({
      file,
      name: file.name,
      status: 'pending'
    }));
    
    setFiles(prev => [...prev, ...newFiles]);
    
    for (const file of acceptedFiles) {
      try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('http://localhost:8000/upload_dataset', {
          method: 'POST',
          body: formData
        });

        if (!response.ok) throw new Error('Upload failed');

        const data = await response.json();
        
        setFiles(prev => prev.map(f => 
          f.name === file.name 
            ? { ...f, status: 'complete' }
            : f
        ));

        setUploadStatus({
          type: 'success',
          message: 'Dataset uploaded successfully!'
        });

        if (onDatasetLoaded) {
          onDatasetLoaded(data);
        }

      } catch (error) {
        setUploadStatus({
          type: 'error',
          message: `Error processing file: ${error.message}`
        });
        
        setFiles(prev => prev.map(f => 
          f.name === file.name 
            ? { ...f, status: 'error' }
            : f
        ));
      }
    }
  };

  const loadHuggingFaceDataset = async () => {
    setLoading(true);
    setUploadStatus({ type: null, message: '' });

    try {
      const response = await fetch('http://localhost:8000/load_huggingface_dataset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_id: hfRepo })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to load dataset');
      }

      const data = await response.json();
      
      setUploadStatus({
        type: 'success',
        message: `Successfully loaded ${data.num_examples} examples from ${hfRepo}`
      });

      if (onDatasetLoaded) {
        onDatasetLoaded(data);
      }

    } catch (error) {
      setUploadStatus({
        type: 'error',
        message: error.message
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Upload Dataset</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="upload">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="upload">File Upload</TabsTrigger>
            <TabsTrigger value="huggingface">HuggingFace</TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="space-y-4">
            <Dropzone
              onChange={processFiles}
              disabled={loading}
              accept={{
                'application/parquet': ['.parquet'],
                'text/csv': ['.csv'],
                'application/json': ['.json']
              }}
            />
            <FileList files={files} />
          </TabsContent>

          <TabsContent value="huggingface">
            <div className="space-y-4">
              <div className="flex gap-2">
                <Input
                  placeholder="e.g. Undi95/orthogonal-activation-steering-TOXIC"
                  value={hfRepo}
                  onChange={(e) => setHfRepo(e.target.value)}
                />
                <Button 
                  onClick={loadHuggingFaceDataset}
                  disabled={loading || !hfRepo}
                >
                  {loading ? (
                    <div className="h-4 w-4 border-2 border-t-white rounded-full animate-spin" />
                  ) : (
                    <Database className="h-4 w-4" />
                  )}
                  <span className="ml-2">Load</span>
                </Button>
              </div>
              <p className="text-sm text-muted-foreground">
                Enter a HuggingFace dataset repository ID to load directly from HuggingFace
              </p>
            </div>
          </TabsContent>
        </Tabs>

        {uploadStatus.message && (
          <Alert 
            variant={uploadStatus.type === 'error' ? 'destructive' : 'default'} 
            className="mt-4"
          >
            <AlertDescription>{uploadStatus.message}</AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default DatasetUploader;

