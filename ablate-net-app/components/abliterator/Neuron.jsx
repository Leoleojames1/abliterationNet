import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

export function Neuron({ position, activation }) {
  const meshRef = useRef()
  const color = new THREE.Color()

  useFrame(() => {
    if (meshRef.current) {
      color.setHSL(activation, 1, 0.5)
      meshRef.current.material.color = color
      meshRef.current.scale.setScalar(0.1 + activation * 0.1)
    }
  })

  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[0.1, 32, 32]} />
      <meshPhongMaterial />
    </mesh>
  )
}

