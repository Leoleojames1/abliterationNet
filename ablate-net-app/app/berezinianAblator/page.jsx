"use client"

import { BerenzinianVisualizer } from '@/components/berezinianAblator/BerenzinianVisualizer'
import { ThemeProvider } from "@/components/theme-provider"

const initialData = {
  layers: [
    [{ id: 1 }, { id: 2 }, { id: 3 }],
    [{ id: 4 }, { id: 5 }, { id: 6 }],
    [{ id: 7 }, { id: 8 }, { id: 9 }],
  ],
  synapses: [
    { source: { x: -2, y: 1, z: 0 }, target: { x: 0, y: 1, z: 0 }, weight: 0.5 },
    { source: { x: -2, y: 0, z: 0 }, target: { x: 0, y: 0, z: 0 }, weight: 0.3 },
    { source: { x: -2, y: -1, z: 0 }, target: { x: 0, y: -1, z: 0 }, weight: 0.7 },
    { source: { x: 0, y: 1, z: 0 }, target: { x: 2, y: 1, z: 0 }, weight: 0.6 },
    { source: { x: 0, y: 0, z: 0 }, target: { x: 2, y: 0, z: 0 }, weight: 0.4 },
    { source: { x: 0, y: -1, z: 0 }, target: { x: 2, y: -1, z: 0 }, weight: 0.8 },
  ],
}

export default function Home() {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <div className="container mx-auto p-4">
        <h1 className="text-4xl font-bold mb-2">Berezinian Abliterator Visualizer</h1>
        <p className="text-xl mb-4">Explore the dynamics of neural networks through the lens of Berezinian algebra</p>
        <BerenzinianVisualizer initialData={initialData} />
      </div>
    </ThemeProvider>
  )
}

