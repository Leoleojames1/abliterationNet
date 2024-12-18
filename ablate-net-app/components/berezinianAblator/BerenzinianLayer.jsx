import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

export function BerenzinianLayer({ neurons, position, isEven }) {
  const groupRef = useRef()

  useFrame(() => {
    if (groupRef.current) {
      groupRef.current.rotation.y += 0.01
    }
  })

  return (
    <group ref={groupRef} position={position}>
      {neurons.map((neuron, i) => {
        const yPos = i - (neurons.length - 1) / 2
        const color = isEven ? new THREE.Color(0x00ff00) : new THREE.Color(0xff0000)
        return (
          <mesh key={i} position={[0, yPos, 0]}>
            <sphereGeometry args={[0.2, 32, 32]} />
            <meshStandardMaterial color={color} />
          </mesh>
        )
      })}
    </group>
  )
}

