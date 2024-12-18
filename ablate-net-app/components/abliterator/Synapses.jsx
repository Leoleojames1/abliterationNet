import { Line, Text } from '@react-three/drei'

export function Synapses({ synapses }) {
  return synapses.map((synapse, i) => (
    <group key={i}>
      <Line 
        points={[synapse.start, synapse.end]}
        color={synapse.weight > 0 ? "cyan" : "magenta"}
        lineWidth={1}
      />
      <Text 
        position={[
          (synapse.start[0] + synapse.end[0]) / 2,
          (synapse.start[1] + synapse.end[1]) / 2,
          (synapse.start[2] + synapse.end[2]) / 2
        ]}
        fontSize={0.1}
        color="white"
      >
        {synapse.weight.toFixed(2)}
      </Text>
    </group>
  ))
}

