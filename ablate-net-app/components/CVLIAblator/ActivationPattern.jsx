import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

export function ActivationPattern({ position, strength }) {
  const meshRef = useRef()

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.x += 0.01
      meshRef.current.rotation.y += 0.01
    }
  })

  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[0.1 + strength * 0.1, 32, 32]} />
      <meshStandardMaterial color={new THREE.Color().setHSL(0.5 + strength * 0.5, 1, 0.5)} />
    </mesh>
  )
}

