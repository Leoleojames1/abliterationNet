"use client"

import { useState, useEffect } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { NeuronLayer } from './NeuronLayer'
import { Synapses } from './Synapses'
import { Button } from "@/components/ui/button"
import { useTheme } from "next-themes"

export function NeuralNetworkVisualizer({ initialData }) {
  const [networkData, setNetworkData] = useState(initialData)
  const { theme, setTheme } = useTheme()

  useEffect(() => {
    setNetworkData(initialData)
  }, [initialData])

  return (
    <div className="w-full h-full bg-gray-900 text-white">
      <Canvas camera={{ position: [0, 0, 10], fov: 75 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />
        {networkData.layers.map((layer, i) => (
          <NeuronLayer key={i} neurons={layer} position={[i * 2 - (networkData.layers.length - 1), 0, 0]} />
        ))}
        <Synapses synapses={networkData.synapses} />
        <OrbitControls />
      </Canvas>
      <div className="absolute top-4 right-4">
        <Button onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
          Toggle Theme
        </Button>
      </div>
    </div>
  )
}

