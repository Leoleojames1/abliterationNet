"use client"

import { CVLIAblatorVisualizer } from '@/components/CVLIAblator/CVLIAblatorVisualizer'
import { ThemeProvider } from "@/components/theme-provider"

export default function Home() {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <div className="container mx-auto p-4">
        <h1 className="text-4xl font-bold mb-2">CVLI Ablator Visualizer</h1>
        <p className="text-xl mb-4">Explore the Contour Vector Line Integral Ablation technique</p>
        <CVLIAblatorVisualizer />
      </div>
    </ThemeProvider>
  )
}

