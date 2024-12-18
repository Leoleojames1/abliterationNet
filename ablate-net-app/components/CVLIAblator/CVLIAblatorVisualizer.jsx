"use client"

import { useState, useEffect } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { ContourPath } from './ContourPath'
import { ActivationPattern } from './ActivationPattern'
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

function generateRandomContour(numPoints = 100) {
  return Array.from({ length: numPoints }, () => [
    (Math.random() - 0.5) * 4,
    (Math.random() - 0.5) * 4,
    (Math.random() - 0.5) * 4
  ])
}

function generateRandomActivations(numActivations = 20) {
  return Array.from({ length: numActivations }, () => ({
    position: [
      (Math.random() - 0.5) * 4,
      (Math.random() - 0.5) * 4,
      (Math.random() - 0.5) * 4
    ],
    strength: Math.random()
  }))
}

export function CVLIAblatorVisualizer() {
  const [contourPath, setContourPath] = useState(generateRandomContour())
  const [activations, setActivations] = useState(generateRandomActivations())
  const [ablationStrength, setAblationStrength] = useState(1)
  const [integrationMethod, setIntegrationMethod] = useState('trapezoidal')

  useEffect(() => {
    const interval = setInterval(() => {
      setActivations(generateRandomActivations())
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  const handleRegenerate = () => {
    setContourPath(generateRandomContour())
    setActivations(generateRandomActivations())
  }

  return (
    <div className="w-full h-screen bg-background text-foreground flex">
      <div className="w-3/4 h-full relative">
        <Canvas camera={{ position: [0, 0, 10], fov: 75 }}>
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} />
          <ContourPath points={contourPath} />
          {activations.map((activation, index) => (
            <ActivationPattern key={index} {...activation} />
          ))}
          <OrbitControls />
        </Canvas>
      </div>
      <div className="w-1/4 h-full p-4 space-y-4 overflow-y-auto">
        <Card>
          <CardHeader>
            <CardTitle>CVLI Ablator Controls</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="ablation-strength">Ablation Strength</Label>
              <Slider
                id="ablation-strength"
                min={0}
                max={2}
                step={0.1}
                value={[ablationStrength]}
                onValueChange={([value]) => setAblationStrength(value)}
              />
            </div>
            <div className="space-y-2">
              <Label>Integration Method</Label>
              <div className="flex space-x-2">
                {['trapezoidal', 'simpson', 'rectangular'].map((method) => (
                  <Button
                    key={method}
                    variant={integrationMethod === method ? "default" : "outline"}
                    onClick={() => setIntegrationMethod(method)}
                  >
                    {method}
                  </Button>
                ))}
              </div>
            </div>
            <Button onClick={handleRegenerate} className="w-full">
              Regenerate Data
            </Button>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Visualization Key</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex items-center justify-between">
              <span>Green Line</span>
              <span className="font-semibold">Contour Path</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Colored Spheres</span>
              <span className="font-semibold">Activation Patterns</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Sphere Size</span>
              <span className="font-semibold">Activation Strength</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Sphere Color</span>
              <span className="font-semibold">Activation Value</span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

