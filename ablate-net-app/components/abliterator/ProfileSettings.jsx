import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';

const ProfileSettings = () => {
  const [apiKey, setApiKey] = useState('');
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    // Here you would typically save the API key to a secure storage
    // For now, we'll just simulate saving
    localStorage.setItem('hf_api_key', apiKey);
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  return (
    <Card className="mt-4">
      <CardHeader>
        <CardTitle>Profile Settings</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <label htmlFor="hf-api-key" className="block text-sm font-medium text-gray-700">
              Hugging Face API Key
            </label>
            <Input
              id="hf-api-key"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter your Hugging Face API key"
            />
          </div>
          <Button onClick={handleSave}>Save API Key</Button>
          {saved && (
            <Alert>
              <AlertDescription>API key saved successfully!</AlertDescription>
            </Alert>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default ProfileSettings;

