import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"

export function AnalysisPanel({ networkData, berezinianWeight }) {
  const [averageWeight, setAverageWeight] = useState(0)
  const [maxWeight, setMaxWeight] = useState(0)
  const [complexity, setComplexity] = useState(0)

  useEffect(() => {
    const weights = networkData.synapses.map(s => s.weight)
    setAverageWeight(weights.reduce((a, b) => a + b, 0) / weights.length)
    setMaxWeight(Math.max(...weights))
    setComplexity(networkData.layers.length * networkData.layers[0].length * berezinianWeight)
  }, [networkData, berezinianWeight])

  return (
    <Card className="w-64">
      <CardHeader>
        <CardTitle>Network Analysis</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <div className="flex justify-between mb-1">
            <span>Average Weight</span>
            <span>{averageWeight.toFixed(2)}</span>
          </div>
          <Progress value={averageWeight * 100} />
        </div>
        <div>
          <div className="flex justify-between mb-1">
            <span>Max Weight</span>
            <span>{maxWeight.toFixed(2)}</span>
          </div>
          <Progress value={maxWeight * 50} />
        </div>
        <div>
          <div className="flex justify-between mb-1">
            <span>Complexity</span>
            <span>{complexity.toFixed(2)}</span>
          </div>
          <Progress value={Math.min(complexity, 100)} />
        </div>
      </CardContent>
    </Card>
  )
}

