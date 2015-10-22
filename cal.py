from struct import unpack

VCGT_TAG = 0x76636774
MLUT_TAG = 0x6d4c5554

class LEDCalibration():
    def __init__(self, filename):
        self.ramp_r = []
        self.ramp_g = []
        self.ramp_b = []

        self.read_vcgt_internal(filename, 256)

        self.ramp_r = list(int(round(x / 256)) for x in self.ramp_r)
        self.ramp_g = list(int(round(x / 256)) for x in self.ramp_g)
        self.ramp_b = list(int(round(x / 256)) for x in self.ramp_b)

    def read_vcgt_internal(self, filename, nEntries):
        with open(filename, "rb") as f:
            f.seek(128, 0)

            numTags = unpack(">I", f.read(4))[0]

            for i in range(0, numTags):
                (tagName, tagOffset, tagSize) = unpack(">III", f.read(3 * 4))

                if tagName == MLUT_TAG:
                    f.seek(tagOffset, 0)
                    print("mLUT found (Profile Mechanic)")
                    print("MLUT not supported yet")
                    break

                if tagName == VCGT_TAG:
                    f.seek(tagOffset, 0)
                    print("vcgt found")
                    tagName = unpack(">I", f.read(4))[0]

                    if tagName != VCGT_TAG:
                        print("invalid content of table vcgt, starting with %x" % tagName)
                        break

                    f.seek(4, 1)
                    gammaType = unpack(">I", f.read(4))[0]

                    if gammaType == 0:
                        (numChannels, numEntries, entrySize) = unpack(">HHH", f.read(3 * 2))

                        print("channels: %d, entry size: %d, entries: %d, tag size: %d" % (numChannels, entrySize, numEntries, tagSize))

                        if numChannels != 3:
                            break # Assume we have always RGB


                        redRamp = unpack(">" + ("H" if entrySize == 2 else "B") * numEntries, f.read(entrySize * numEntries))
                        greenRamp = unpack(">" + ("H" if entrySize == 2 else "B") * numEntries, f.read(entrySize * numEntries))
                        blueRamp = unpack(">" + ("H" if entrySize == 2 else "B") * numEntries, f.read(entrySize * numEntries))

                        if numEntries >= nEntries:
                            ratio = int(numEntries / nEntries)

                            for j in range(0, nEntries):
                                self.ramp_r.append(redRamp[ratio * j])
                                self.ramp_g.append(greenRamp[ratio * j])
                                self.ramp_b.append(blueRamp[ratio * j])

                        else:
                            # add extrapolated upper limit to the arrays - handle overflow
                            redRamp.append((2 * redRamp[-1] - redRamp[-2] ) & 0xffff)
                            if redRamp[numEntries] < 0x4000:
                                redRamp[numEntries] = 0xffff

                            greenRamp.append((2 * greenRamp[-1] - greenRamp[-2] ) & 0xffff)
                            if greenRamp[numEntries] < 0x4000:
                                greenRamp[numEntries] = 0xffff

                            blueRamp.append((2 * blueRamp[-1] - blueRamp[-2] ) & 0xffff)
                            if blueRamp[numEntries] < 0x4000:
                                blueRamp[numEntries] = 0xffff

                            print("Interpolated ramps not yet supported")
                            break

                    else:
                        print("Unsupported gamma type %d" % gammaType)

    def get_rgb(self, r, g, b):
        r = min(255, int(r))
        g = min(255, int(g))
        b = min(255, int(b))
        return (self.ramp_r[r], self.ramp_g[g], self.ramp_b[b])

if __name__ == '__main__':
    c = LEDCalibration("ws2812b-9500.icc")

