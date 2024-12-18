import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

export function VisualizerKey({ isEven }) {
  return (
    <Card className="w-64">
      <CardHeader>
        <CardTitle>Visualizer Key</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex items-center justify-between">
          <span>Neurons</span>
          <Badge variant={isEven ? "success" : "destructive"}>
            {isEven ? "Even (Green)" : "Odd (Red)"}
          </Badge>
        </div>
        <div className="flex items-center justify-between">
          <span>Synapses</span>
          <Badge variant="secondary">Color Intensity = Weight</Badge>
        </div>
        <div className="flex items-center justify-between">
          <span>Rotation</span>
          <Badge variant="outline">Continuous</Badge>
        </div>
      </CardContent>
    </Card>
  )
}

