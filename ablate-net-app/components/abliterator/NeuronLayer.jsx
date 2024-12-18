import { Text } from '@react-three/drei'
import { Neuron } from './Neuron'

export function NeuronLayer({ neurons, position }) {
  return (
    <group position={position}>
      {neurons.map((neuron, i) => (
        <Neuron 
          key={i} 
          position={[0, i * 0.5 - (neurons.length - 1) * 0.25, 0]} 
          activation={neuron.activation} 
        />
      ))}
      <Text position={[0, -1, 0]} fontSize={0.2} color="white">
        Layer {position[0] / 2 + (neurons.length / 2)}
      </Text>
    </group>
  )
}

