"use client"

import { useState, useEffect } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { BerenzinianLayer } from './BerenzinianLayer'
import { Synapses } from './Synapses'
import { Button } from "@/components/ui/button"
import { useTheme } from "next-themes"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { VisualizerKey } from './VisualizerKey'
import { AnalysisPanel } from './AnalysisPanel'
import { Card, CardContent } from "@/components/ui/card"

export function BerenzinianVisualizer({ initialData }) {
  const [networkData, setNetworkData] = useState(initialData)
  const [isEven, setIsEven] = useState(true)
  const [berezinianWeight, setBerezinianWeight] = useState(1)
  const { theme, setTheme } = useTheme()

  useEffect(() => {
    const interval = setInterval(() => {
      // Simulate data updates
      setNetworkData(prevData => ({
        ...prevData,
        layers: prevData.layers.map(layer => 
          layer.map(neuron => ({ ...neuron, activation: Math.random() }))
        ),
        synapses: prevData.synapses.map(synapse => ({
          ...synapse,
          weight: Math.min(Math.max(synapse.weight + (Math.random() - 0.5) * 0.1 * berezinianWeight, 0), 1)
        }))
      }))
    }, 1000)

    return () => clearInterval(interval)
  }, [berezinianWeight])

  return (
    <div className="w-full h-screen bg-background text-foreground flex">
      <div className="w-3/4 h-full relative">
        <Canvas camera={{ position: [0, 0, 10], fov: 75 }}>
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} />
          {networkData.layers.map((layer, i) => (
            <BerenzinianLayer 
              key={i} 
              neurons={layer} 
              position={[i * 2 - (networkData.layers.length - 1), 0, 0]} 
              isEven={isEven}
            />
          ))}
          <Synapses synapses={networkData.synapses} berezinianWeight={berezinianWeight} />
          <OrbitControls />
        </Canvas>
      </div>
      <div className="w-1/4 h-full p-4 space-y-4 overflow-y-auto">
        <Card>
          <CardContent className="p-4 space-y-4">
            <Button onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} className="w-full">
              Toggle Theme
            </Button>
            <div className="flex items-center space-x-2">
              <Switch
                id="even-odd-switch"
                checked={isEven}
                onCheckedChange={setIsEven}
              />
              <Label htmlFor="even-odd-switch">Even/Odd</Label>
            </div>
            <div className="space-y-2">
              <Label htmlFor="berezinian-weight">Berezinian Weight</Label>
              <Slider
                id="berezinian-weight"
                min={0}
                max={2}
                step={0.1}
                value={[berezinianWeight]}
                onValueChange={([value]) => setBerezinianWeight(value)}
              />
            </div>
          </CardContent>
        </Card>
        <VisualizerKey isEven={isEven} />
        <AnalysisPanel networkData={networkData} berezinianWeight={berezinianWeight} />
      </div>
    </div>
  )
}

