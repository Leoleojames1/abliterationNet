import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

export function Synapses({ synapses, berezinianWeight }) {
  const meshRef = useRef()
  
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    const positions = new Float32Array(synapses.length * 6)
    const colors = new Float32Array(synapses.length * 6)

    synapses.forEach((synapse, i) => {
      const { source, target, weight } = synapse
      positions[i * 6] = source.x
      positions[i * 6 + 1] = source.y
      positions[i * 6 + 2] = source.z
      positions[i * 6 + 3] = target.x
      positions[i * 6 + 4] = target.y
      positions[i * 6 + 5] = target.z

      const color = new THREE.Color()
      color.setHSL(0.5 + weight * berezinianWeight * 0.5, 1, 0.5)
      colors[i * 6] = color.r
      colors[i * 6 + 1] = color.g
      colors[i * 6 + 2] = color.b
      colors[i * 6 + 3] = color.r
      colors[i * 6 + 4] = color.g
      colors[i * 6 + 5] = color.b
    })

    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    return geo
  }, [synapses, berezinianWeight])

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.002
    }
  })

  return (
    <lineSegments ref={meshRef} geometry={geometry}>
      <lineBasicMaterial vertexColors attach="material" />
    </lineSegments>
  )
}

